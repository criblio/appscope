#ifndef __STATE_PRIVATE_H__
#define __STATE_PRIVATE_H__

#include <limits.h>
#include <sys/socket.h>

#define PROTOCOL_STR 16
#define FUNC_MAX 24
#define HDRTYPE_MAX 16

//
// This file contains implementation details for state.c and reporting.c.
// It is expected that state.c and report.c will both directly include
// this file, but it is not recommended that any other file include it.
//

typedef enum
{
    DETECT_PENDING, // Initial State, waiting for packet to detect
    DETECT_FALSE,   // Not detected
    DETECT_TRUE     // Detected
} detect_type_t;

typedef enum
{
    INET_TCP,
    INET_UDP,
    UNIX_TCP,
    UNIX_UDP,
    SOCK_OTHER,
    SOCK_NUM_BUCKETS
} sock_summary_bucket_t;

typedef struct {
    uint64_t mtc;
    uint64_t evt;
} counters_element_t;

typedef struct metric_counters_t {
    counters_element_t  openPorts;
    counters_element_t  netConnectionsUdp;
    counters_element_t  netConnectionsTcp;
    counters_element_t  netConnectionsOther;
    counters_element_t  netrxBytes[SOCK_NUM_BUCKETS];
    counters_element_t  nettxBytes[SOCK_NUM_BUCKETS];
    counters_element_t  readBytes;
    counters_element_t  writeBytes;
    counters_element_t  numSeek;
    counters_element_t  numStat;
    counters_element_t  numOpen;
    counters_element_t  numClose;
    counters_element_t  numDNS;
    counters_element_t  fsDurationNum;
    counters_element_t  fsDurationTotal;
    counters_element_t  connDurationNum;
    counters_element_t  connDurationTotal;
    counters_element_t  dnsDurationNum;
    counters_element_t  dnsDurationTotal;
    counters_element_t  netConnOpen;
    counters_element_t  netConnClose;
    counters_element_t  netConnectErrors;
    counters_element_t  netTxRxErrors;
    counters_element_t  netDNSErrors;
    counters_element_t  fsOpenCloseErrors;
    counters_element_t  fsRdWrErrors;
    counters_element_t  fsStatErrors;
} metric_counters;

typedef struct {
    struct {
        int open_close;
        int read_write;
        int stat;
        int seek;
        int error;
    } fs;
    struct {
        int open_close;
        int rx_tx;
        int dns;
        int error;
        int dnserror;
    } net;
} summary_t;

typedef struct evt_type_t {
    metric_t evtype;
} evt_type;

typedef struct http_post_t {
    int ssl;
    uint64_t start_duration;
    uint64_t id;
    char *hdr;
} http_post;

typedef struct http_map_t {
    time_t first_time;
    uint64_t start_time;
    uint64_t duration;
    uint64_t id;
    char *req;          // The whole original request
    size_t req_len;
    char *method_str;   //   Method field from Request-Line
    char *target_str;   //   Request-URI field from Request-Line
    size_t clen;        //   Content-Length entity-header value from req
    char *resp;         // The whole original response
} http_map;

typedef struct stat_err_info_t {
    metric_t evtype;
    metric_t data_type;
    char name[PATH_MAX];
    char funcop[FUNC_MAX];
    metric_counters counters;
} stat_err_info;

typedef enum {
    HTTP_NONE,
    HTTP_HDR,
    HTTP_HDREND,
    HTTP_DATA
} http_enum_t;

typedef struct
{
    uint64_t uid;
    int sockfd;
    int isSsl;
    metric_t src;
} httpId_t;

// storage for partial HTTP/2 frames
typedef struct {
    uint8_t *buf;  // bytes array pointer
    size_t   len;  // num bytes used
    size_t   size; // num bytes allocated
} http_buf_t;

typedef struct {
    metric_t evtype;
    char lib[PATH_MAX];
    char path[PATH_MAX];
    char host[INET6_ADDRSTRLEN];
    char func[FUNC_MAX];
} security_info_t;

typedef struct protocol_info_t {
    metric_t evtype;
    metric_t ptype;
    int isServer;
    size_t len;
    int fd;
    uint64_t uid;
    char *data;
    int sock_type;
    struct sockaddr_storage localConn;
    struct sockaddr_storage remoteConn;
} protocol_info;

typedef enum {HTTP_RX, HTTP_TX, HTTP_NUM} http_direction_t;

typedef struct {
    http_enum_t state;
    char *hdr;          // Used if state == HDR
    size_t hdrlen;
    size_t hdralloc;
    size_t clen;        // Used if state==HTTP_DATA
    httpId_t id;

    // HTTP version detected (0=unknown, 1=HTTP/1.x, 2=HTTP/2.0)
    int version;

    // These will be TRUE if `hdr` ...
    bool isResponse;           // ... contains a complete response header
    bool hasUpgrade;           // ... has `Upgrade: h2c` header
    bool hasConnectionUpgrade; // ... has `Connection: upgrade` header

    // HTTP/2 state
    http_buf_t http2Buf; // buffers for partial frames
} http_state_t;

typedef struct net_info_t {
    metric_t evtype;
    metric_t data_type;
    int fd;
    int active;
    int type;
    http_state_t http[HTTP_NUM];  // rx=[0] and tx=[1]
    bool urlRedirect;
    bool addrSetLocal;
    bool addrSetRemote;
    bool addrSetUnix;
    bool remoteClose;
    counters_element_t numTX;
    counters_element_t numRX;
    counters_element_t txBytes;
    counters_element_t rxBytes;
    bool dnsSend;
    bool dnsRecv;
    uint64_t startTime;
    counters_element_t numDuration;
    counters_element_t totalDuration;
    uint64_t uid;
    uint64_t lnode;
    uint64_t rnode;
    char dnsName[MAX_HOSTNAME];
    cJSON *dnsAnswer;
    struct sockaddr_storage localConn;
    struct sockaddr_storage remoteConn;
    metric_counters counters;

    detect_type_t tlsDetect;     // state for TLS detection on this channel
    protocol_def_t* tlsProtoDef; // the TLS protocol-detector used

    detect_type_t protoDetect;     // state for protocol detection on this channel
    protocol_def_t* protoProtoDef; // The protocol-detector that matched

} net_info;

typedef struct fs_info_t {
    metric_t evtype;
    metric_t data_type;
    int fd;
    int active;
    fs_content_type_t content_type;
    fs_type_t type;
    counters_element_t numOpen;
    counters_element_t numClose;
    counters_element_t numSeek;
    counters_element_t numRead;
    counters_element_t numWrite;
    counters_element_t readBytes;
    counters_element_t writeBytes;
    counters_element_t numDuration;
    counters_element_t totalDuration;
    uint64_t uid;
    uid_t fuid;
    gid_t fgid;
    mode_t mode;
    char path[PATH_MAX];
    char funcop[FUNC_MAX];
} fs_info;

typedef struct payload_info_t {
    metric_t evtype;
    metric_t src;
    int sockfd;
    net_info net;
    size_t len;
    char *data;
} payload_info;

// Accessor functions defined in state.c, but used in report.c too.
int get_port(int, int, control_type_t);
int get_port_net(net_info *, int, control_type_t);
bool checkNetEntry(int);
bool checkFSEntry(int);
net_info *getNetEntry(int);
fs_info *getFSEntry(int);
bool addrIsNetDomain(struct sockaddr_storage *);
bool addrIsUnixDomain(struct sockaddr_storage *);
sock_summary_bucket_t getNetRxTxBucket(net_info *);

// The hiding of objects forces these to be defined here
void doFSMetric(metric_t, struct fs_info_t *, control_type_t, const char *, ssize_t, const char *);
void doNetMetric(metric_t, struct net_info_t *, control_type_t, ssize_t);
void doUnixEndpoint(int, net_info *);
void resetInterfaceCounts(counters_element_t *);
void addToInterfaceCounts(counters_element_t *, uint64_t);
void subFromInterfaceCounts(counters_element_t *, uint64_t);

// Data that lives in state.c, but is used in report.c too.
extern summary_t g_summary;
extern net_info *g_netinfo;
extern fs_info *g_fsinfo;
extern metric_counters g_ctrs;

#endif // __STATE_PRIVATE_H__
