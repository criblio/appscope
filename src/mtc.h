#ifndef __MTC_H__
#define __MTC_H__
#include "log.h"
#include "mtcformat.h"
#include "transport.h"

typedef struct _mtc_t mtc_t;

// Constructors Destructors
mtc_t *mtcCreate();
void mtcDestroy(mtc_t **);

// Accessors
unsigned mtcEnabled(mtc_t *);
int mtcSend(mtc_t *, const char *msg);
int mtcSendMetric(mtc_t *, event_t *);
void mtcFlush(mtc_t *);

// Setters (modifies mtc_t, but does not persist modifications)
int mtcNeedsConnection(mtc_t *);
int mtcConnect(mtc_t *);
int mtcConnection(mtc_t *);
int mtcDisconnect(mtc_t *);
int mtcReconnect(mtc_t *);
void mtcEnabledSet(mtc_t *, unsigned);
void mtcTransportSet(mtc_t *, transport_t *);
void mtcFormatSet(mtc_t *, mtc_fmt_t *);

#endif // __MTC_H__
