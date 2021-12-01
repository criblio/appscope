/*
 * Using A Netlink socket with the sock_diag module
 *
 * gcc test/manual/socket_diag.c -o sdiag
 */

#include <errno.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sock_diag.h>
#include <linux/unix_diag.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int
send_query(int fd, unsigned long inode)
{
    struct sockaddr_nl nladdr = {.nl_family = AF_NETLINK};
    struct {
        struct nlmsghdr nlh;
        struct unix_diag_req udr;
    } req = {.nlh =
                 {
                     .nlmsg_len = sizeof(req),
                     .nlmsg_type = SOCK_DIAG_BY_FAMILY,
                     .nlmsg_flags = NLM_F_REQUEST //| NLM_F_DUMP
                 },
             .udr = {
                 .sdiag_family = AF_UNIX,
                 .sdiag_protocol = 0,
                 .pad = 0,
                 .udiag_states = -1,
                 .udiag_ino = inode,
                 .udiag_cookie[0] = -1,        //~0U,
                 .udiag_cookie[1] = -1,        //~0U,
                 .udiag_show = UDIAG_SHOW_PEER // UDIAG_SHOW_NAME | UDIAG_SHOW_PEER
             }};
    struct iovec iov = {.iov_base = &req, .iov_len = sizeof(req)};
    struct msghdr msg = {.msg_name = (void *)&nladdr, .msg_namelen = sizeof(nladdr), .msg_iov = &iov, .msg_iovlen = 1};

    for (;;) {
        if (sendmsg(fd, &msg, 0) < 0) {
            if (errno == EINTR)
                continue;

            perror("sendmsg");
            return -1;
        }

        return 0;
    }
}

static int
print_diag(const struct unix_diag_msg *diag, unsigned int len)
{
    if (len < NLMSG_LENGTH(sizeof(*diag))) {
        fputs("short response\n", stderr);
        return -1;
    }
    if (diag->udiag_family != AF_UNIX) {
        fprintf(stderr, "unexpected family %u\n", diag->udiag_family);
        return -1;
    }

    struct rtattr *attr;
    unsigned int rta_len = len - NLMSG_LENGTH(sizeof(*diag));
    unsigned int peer = 0;
    size_t path_len = 0;
    char path[sizeof(((struct sockaddr_un *)0)->sun_path) + 1];

    for (attr = (struct rtattr *)(diag + 1); RTA_OK(attr, rta_len); attr = RTA_NEXT(attr, rta_len)) {
        switch (attr->rta_type) {
            case UNIX_DIAG_NAME:
                if (!path_len) {
                    path_len = RTA_PAYLOAD(attr);
                    if (path_len > sizeof(path) - 1)
                        path_len = sizeof(path) - 1;
                    memcpy(path, RTA_DATA(attr), path_len);
                    path[path_len] = '\0';
                }
                break;

            case UNIX_DIAG_PEER:
                if (RTA_PAYLOAD(attr) >= sizeof(peer))
                    peer = *(unsigned int *)RTA_DATA(attr);
                break;
        }
    }

    printf("inode=%u", diag->udiag_ino);

    if (peer)
        printf(", peer=%u", peer);

    if (path_len)
        printf(", name=%s%s", *path ? "" : "@", *path ? path : path + 1);

    putchar('\n');
    return 0;
}

static int
receive_responses(int fd)
{
    long buf[8192 / sizeof(long)];
    struct sockaddr_nl nladdr = {.nl_family = AF_NETLINK};
    struct iovec iov = {.iov_base = buf, .iov_len = sizeof(buf)};
    int flags = 0;

    for (;;) {
        struct msghdr msg = {.msg_name = (void *)&nladdr, .msg_namelen = sizeof(nladdr), .msg_iov = &iov, .msg_iovlen = 1};

        ssize_t ret = recvmsg(fd, &msg, flags);

        if (ret < 0) {
            if (errno == EINTR)
                continue;

            perror("recvmsg");
            return -1;
        }
        if (ret == 0)
            return 0;

        const struct nlmsghdr *h = (struct nlmsghdr *)buf;

        if (!NLMSG_OK(h, ret)) {
            fputs("!NLMSG_OK\n", stderr);
            return -1;
        }

        for (; NLMSG_OK(h, ret); h = NLMSG_NEXT(h, ret)) {
            if (h->nlmsg_type == NLMSG_DONE)
                return 0;

            if (h->nlmsg_type == NLMSG_ERROR) {
                const struct nlmsgerr *err = NLMSG_DATA(h);

                if (h->nlmsg_len < NLMSG_LENGTH(sizeof(*err))) {
                    fputs("NLMSG_ERROR\n", stderr);
                } else {
                    errno = -err->error;
                    perror("NLMSG_ERROR");
                }

                return -1;
            }

            if (h->nlmsg_type != SOCK_DIAG_BY_FAMILY) {
                fprintf(stderr, "unexpected nlmsg_type %u\n", (unsigned)h->nlmsg_type);
                return -1;
            }

            if (print_diag(NLMSG_DATA(h), h->nlmsg_len))
                return -1;
        }
    }
}

static unsigned long
get_response(int fd)
{
    long buf[sizeof(struct nlmsghdr) + sizeof(long)];
    struct unix_diag_msg *diag;
    struct rtattr *attr;
    struct sockaddr_nl nladdr = {.nl_family = AF_NETLINK};
    struct iovec iov = {.iov_base = buf, .iov_len = sizeof(buf)};
    int flags = 0;

    struct msghdr msg = {.msg_name = (void *)&nladdr, .msg_namelen = sizeof(nladdr), .msg_iov = &iov, .msg_iovlen = 1};

    ssize_t ret = recvmsg(fd, &msg, flags);

    if (ret < 0) {
        perror("recvmsg");
        return -1;
    }

    if (ret == 0)
        return 0;

    const struct nlmsghdr *h = (struct nlmsghdr *)buf;

    if (!NLMSG_OK(h, ret)) {
        fputs("!NLMSG_OK\n", stderr);
        return -1;
    }

    for (; NLMSG_OK(h, ret); h = NLMSG_NEXT(h, ret)) {
        if (h->nlmsg_type == NLMSG_DONE)
            return 0;

        if (h->nlmsg_type == NLMSG_ERROR) {
            const struct nlmsgerr *err = NLMSG_DATA(h);

            if (h->nlmsg_len < NLMSG_LENGTH(sizeof(*err))) {
                fputs("NLMSG_ERROR\n", stderr);
            } else {
                errno = -err->error;
                perror("NLMSG_ERROR");
            }

            return -1;
        }

        if (h->nlmsg_type != SOCK_DIAG_BY_FAMILY) {
            fprintf(stderr, "unexpected nlmsg_type %u\n", (unsigned)h->nlmsg_type);
            return -1;
        }

        if ((diag = NLMSG_DATA(h)) != NULL) {
            if (h->nlmsg_len < NLMSG_LENGTH(sizeof(*diag))) {
                fputs("short response\n", stderr);
                return -1;
            }

            if (diag->udiag_family != AF_UNIX) {
                fprintf(stderr, "unexpected family %u\n", diag->udiag_family);
                return -1;
            }

            attr = (struct rtattr *)(diag + 1);
            if (attr->rta_type == UNIX_DIAG_PEER) {
                if (RTA_PAYLOAD(attr) >= sizeof(unsigned int)) {
                    return *(unsigned int *)RTA_DATA(attr);
                }
            }
        }
    }
    return 0;
}

int
main(int argc, char **argv)
{
    unsigned long inode;

    if (argc > 1) {
        inode = atol(argv[1]);
    } else {
        inode = 0;
    }

    int fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_SOCK_DIAG);

    if (fd < 0) {
        perror("socket");
        return 1;
    }

    printf("%s:%d %ld\n", __FUNCTION__, __LINE__, inode);
    send_query(fd, inode);
    // receive_responses(fd, inode);
    inode = get_response(fd);
    printf("%s:%d %ld\n", __FUNCTION__, __LINE__, inode);

    close(fd);
    return 0;
}
