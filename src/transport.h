#ifndef __TRANSPORT_H__
#define __TRANSPORT_H__

typedef struct _transport_t transport_t;

// Constructors Destructors
transport_t*        transportCreateUdp(char* host, int port);
transport_t*        transportCreateFile(char* path);
transport_t*        transportCreateUnix(char* path);
transport_t*        transportCreateSyslog(void);
transport_t*        transportCreateShm();
void                transportDestroy(transport_t**);

// Accessors
int                 transportSend(transport_t*, char* msg);

#endif // __TRANSPORT_H__
