#include <stddef.h>
#include "transport.h"

typedef enum {TX_UDP, TX_UNIX, TX_FILE, TX_SYSLOG, TX_SHM} tx_t;

struct _transport_t
{
    tx_t type;
    
};

transport_t*
transportCreateUdp(char* host, int port)
{
    return NULL;
}

transport_t*
transportCreateUnix(char* path)
{
    return NULL;
}

transport_t*
transportCreateFile(char* path)
{
    return NULL;
}

transport_t*
transportCreateSyslog(void)
{
    return NULL;
}

transport_t*
transportCreateShm()
{
    return NULL;
}

void
transportDestroy(transport_t** transport)
{

}

int
transportSend(transport_t* transport, char* msg)
{
    return -1;
}


/*

// These need to come from a config file 
#define LOG_FILE 1  // eventually an enum for file, syslog, shared memory  
static bool g_log = TRUE; 
static const char g_logFile[] = "/tmp/scope.log"; 
static unsigned int g_logOp = LOG_FILE; 
static int g_logfd = -1;

void scopeLog(char *msg, int fd)
{
    size_t len;

    if ((g_log == FALSE) || (!msg)) {
        return;
    }

    if (g_logOp & LOG_FILE) {
        char buf[strlen(msg) + 128];

        if ((g_logfd == -1) &&
            (strlen(g_logFile) > 0)) {
                g_logfd = open(g_logFile, O_RDWR|O_APPEND);
        }

        len = sizeof(buf) - strlen(buf);
        snprintf(buf, sizeof(buf), "Scope: %s(%d): ", g_procname, fd);
        strncat(buf, msg, len);
        g_fn.write(g_logfd, buf, strlen(buf));
    }
}

static int g_sock = 0;
static struct sockaddr_in g_saddr;
static operations_info g_ops;

static
void initSocket(config_t* cfg)
{
    int flags;

    // JRC TBD: We eventually need to support UNIX, FILE, and SYSLOG too...
    if (cfgOutTransportType(cfg) != CFG_UDP) {
        scopeLog("initSocket: unsupported TransportType\n", -1);
        return;
    }

    // Create a UDP socket
    g_sock = g_fn.socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0)     {
        scopeLog("ERROR: initSocket:socket\n", -1);
    }

    // Set the socket to non blocking
    flags = fcntl(g_sock, F_GETFL, 0);
    fcntl(g_sock, F_SETFL, flags | O_NONBLOCK);

    // Create the address to send to
    memset(&g_saddr, 0, sizeof(g_saddr));
    g_saddr.sin_family = AF_INET;
    g_saddr.sin_port = htons(cfgOutTransportPort(cfg));
    if (inet_aton(cfgOutTransportHost(cfg), &g_saddr.sin_addr) == 0) {
        scopeLog("ERROR: initSocket:inet_aton\n", -1);
    }
}

static
void postMetric(const char *metric)
{
    ssize_t rc;

    if (!g_fn.socket) {
        // initSocket must have failed during the constructor
        scopeLog("postMetric: uninitialized socket\n", -1);
        return;
    }

    scopeLog((char *)metric, -1);
    if (g_fn.sendto) {
        rc = g_fn.sendto(g_sock, metric, strlen(metric), 0,
                         (struct sockaddr *)&g_saddr, sizeof(g_saddr));
        if (rc < 0) {
            scopeLog("ERROR: sendto\n", g_sock);
            switch (errno) {
            case EWOULDBLOCK:
                g_ops.udp_blocks++;
                break;
            default:
                g_ops.udp_errors++;
            }
        }
    }
}

*/
