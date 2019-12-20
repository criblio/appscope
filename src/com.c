#include "com.h"


static int
postMsg(ctl_t *ctl, char *msg, upload_type_t type, request_t *req, bool now)
{
    int rc = -1;
    char *streamMsg;
    upload_t upld;

    upld.type = type;
    upld.msg = msg;
    upld.body = NULL;
    upld.req = req;
    streamMsg = ctlCreateTxMsg(&upld);

    if (streamMsg) {
        // on the ring buffer
        ctlSendMsg(ctl, streamMsg);

        // send it now or periodic
        if (now) ctlFlush(ctl);
        rc = 0;
    }

    return rc;
}

int
cmdPostEvtMsg(ctl_t *ctl, char *msg)
{
    return postMsg(ctl, msg, UPLD_EVT, NULL, FALSE);
}

int
cmdPostInfoMsg(ctl_t *ctl, char *msg)
{
    return postMsg(ctl, msg, UPLD_INFO, NULL, FALSE);
}

int
cmdSendEvtMsg(ctl_t *ctl, char *msg)
{
    return postMsg(ctl, msg, UPLD_EVT, NULL, TRUE);
}

int
cmdSendInfoMsg(ctl_t *ctl, char *msg)
{
    return postMsg(ctl, msg, UPLD_INFO, NULL, TRUE);
}

int
cmdSendResponse(ctl_t *ctl, request_t *req)
{
    return postMsg(ctl, NULL, UPLD_RESP, req, TRUE);
}

request_t *
cmdParse(ctl_t *ctl, char *cmd)
{
    return ctlParseRxMsg((const char *)cmd);
}

char *
msgStart(rtconfig *rcfg, config_t *scfg)
{
    return jsonStringFromCfg(scfg);
}

char *
msgEvtMetric(evt_t *evt, event_t *metric, uint64_t uid, rtconfig *rcfg)
{
    return evtMetric(evt, rcfg->hostname, rcfg->cmd, rcfg->procname, uid, metric);
}

char *
msgEvtLog(evt_t *evt, const char *path, const void *buf, size_t len,
          uint64_t uid, rtconfig *rcfg)
{
    return evtLog(evt, rcfg->hostname, path, rcfg->cmd, rcfg->procname,
                  buf, len, uid);

}

