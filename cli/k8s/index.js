const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const https = require('https');
const jsonpatch = require('fast-json-patch');


function sendResponse(req, res, rest) {
    const resp = {
        uid: req.body.request.uid,
        allowed: true,
        ...rest
    };

    res.type('application/json');
    res.status(200).send({
        kind: req.body.kind,
        apiVersion: req.body.apiVersion,
        request: req.body.request,
        response: resp
    });
}

function sendError(req, res, errText) {
    console.log(`error: ${errText}`);
    sendResponse(req, res, {});
}

// Returns zero or more operations to do the passed array
// Checks if the array exists first, creates an additional add operation if it does not
function addToArray(obj, arrayMember) {
    let ret = [];
    if (typeof(arrayMember) !== 'object') return ret;
    let path = arrayMember.path;
    if (!path) return ret;
    path = path.replace(/\/-$/, ''); // Trim last /-
    const val = jsonpatch.getValueByPointer(obj, path)
    if (!val) {
        ret.push({
            op: 'add',
            path: path,
            value: []
        })
    }
    ret.push(arrayMember);
    return ret;
}


const app = express()
app.use(bodyParser.json())

app.post('/mutate', (req, res, next) => {
    let admReq = req.body.request;
    console.log(admReq.uid + ' - ' + admReq.resource.resource + ' - ' + admReq.name + ' - ' + admReq.namespace + ' - ' + admReq.operation);
    console.log(JSON.stringify(req.body));
    console.log('Validating request');
    let shouldModify = true;
    if (req.body.kind !== 'AdmissionReview') return sendError(req, res, 'kind is not AdmissionReview');
    if (!admReq.object) return sendError(req, res, 'object missing from AdmissionReview');
    if (!admReq.object.metadata) return sendError(req, res, 'metadata missing from request.object');
    if (!admReq.object.metadata.labels) return sendError(req, res, 'labels missing from request.object.metadata');
    if (!admReq.object.spec.containers) return sendError(req, res, 'containers missing from request.object.spec');
    if (admReq.object.metadata.annotations) {
        if (admReq.object.metadata.annotations['io.cribl.scope/disable']) {
            shouldModify = false;
        }
    }
    if (!shouldModify) return sendResponse(req, res, {});

    let jsonPatch = [
        ...addToArray(admReq.object, {
            op: 'add',
            path: '/spec/initContainers/-',
            value: {
                name: 'scope-init',
                image: 'cribl/scope-init:0.0.4',
                volumeMounts: [{
                    mountPath: '/scope',
                    name: 'scope',
                }]
            },
        }),
        ...addToArray(admReq.object, {
            op: 'add',
            path: '/spec/volumes/-',
            value: {
                emptyDir: {},
                name: 'scope',
            },
        })
    ];
    for (let i = 0; i < admReq.object.spec.containers.length; i++) {
        jsonPatch = jsonPatch.concat(addToArray(admReq.object, {
            op: 'add',
            path: `/spec/containers/${i}/volumeMounts/-`,
            value: {
                mountPath: '/scope',
                name: 'scope',
            }
        }));
        jsonPatch = jsonPatch.concat(addToArray(admReq.object, {
            op: 'add',
            path: `/spec/containers/${i}/env/-`,
            value: {
                name: 'LD_PRELOAD',
                value: '/scope/libscope.so',
            },
        }));
        // jsonPatch = jsonPatch.concat(addToArray(admReq.object, {
        //     op: 'add',
        //     path: `/spec/containers/$[i}/command/0`,
        //     value: '/scope/scope',
        // }));
        jsonPatch.push({
            op: 'add',
            path: `/spec/containers/${i}/env/-`,
            value: {
                name: 'SCOPE_CONF_PATH',
                value: '/scope/scope.yml',
            },
        });
        jsonPatch.push({
            op: 'add',
            path: `/spec/containers/${i}/env/-`,
            value: {
                name: 'SCOPE_EXEC_PATH',
                value: '/scope/scope',
            },
        });
        jsonPatch.push({
            op: 'add',
            path: `/spec/containers/${i}/env/-`,
            value: {
                name: 'SCOPE_TAG_node_name',
                valueFrom: {
                    fieldRef: {
                        fieldPath: 'spec.nodeName',
                    },
                },
            },
        });
        jsonPatch.push({
            op: 'add',
            path: `/spec/containers/${i}/env/-`,
            value: {
                name: 'SCOPE_TAG_pod_name',
                valueFrom: {
                    fieldRef: {
                        fieldPath: 'metadata.name',
                    },
                },
            },
        });
        jsonPatch.push({
            op: 'add',
            path: `/spec/containers/${i}/env/-`,
            value: {
                name: 'SCOPE_TAG_namespace',
                valueFrom: {
                    fieldRef: {
                        fieldPath: 'metadata.namespace',
                    },
                },
            },
        });
        Object.keys(admReq.object.metadata.labels).forEach(k => {
            if (k.startsWith('app.kubernetes.io')) {
                const parts = k.split('/');
                if (parts.length > 1) {
                    const name = `SCOPE_TAG_${parts[1].toLowerCase()}`;
                    const value = admReq.object.metadata.labels[k];
                    jsonPatch.push({
                        op: 'add',
                        path: `/spec/containers/${i}/env/-`,
                        value: {
                            name,
                            value,
                        },
                    });
                }
            }
        });
    }
    console.log('JSONPatch: ', JSON.stringify(jsonPatch));

    sendResponse(req, res, {
        patchType: 'JSONPatch',
        patch: new Buffer.from(JSON.stringify(jsonPatch)).toString('base64'),
    });
});

https.createServer({
    key: fs.readFileSync('/etc/certs/tls.key'),
    cert: fs.readFileSync('/etc/certs/tls.crt')
}, app).listen(4443);
