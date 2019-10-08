exports.disabled = 0;
exports.name = 'REST Lookup';
exports.version = 0.1;
exports.group = 'Demo Functions';

const url = require('url');

const dLogger = C.util.getLogger('restLookup');

const { Expression } = C.expr;

let urlExpression;
let http;
let agent;

const maxRetryInterval = 10000;

exports.init = (opt) => {
  conf = (opt || {}).conf || {};

  urlExpression = new Expression(conf.urlExpression, { disallowAssign: true });
  const u = urlExpression.evalOn({});
  const parsedurl = new url.URL(u);
  const proto = parsedurl.protocol.substr(0, parsedurl.protocol.length - 1);
  // Avoid dynamic imports
  const agentOpts = { keepAlive: true, maxSockets: conf.maxSockets || 10 };
  if (proto === 'http') {
    http = require('http');
    agent = http.Agent(agentOpts);
  } else {
    http = require('https');
    agent = https.Agent(agentOpts);
  }

  if (conf.headers && Array.isArray(conf.headers)) {
    conf.headersObj = conf.headers.reduce((prev, cur) => Object.keys(cur).forEach(k => prev[k] = cur[k]), {});
  }

  if (conf.retryOnError === undefined) {
    conf.retryOnError = false;
  }
};

function lookupEvent(event, u, retryInterval, promise) {
  promise = promise || Promise.resolve(); // if promise is undefined, create empty promise

  return promise.then(() => new Promise((resolve, reject) => {
    const onError = (err) => {
      dLogger.error(`Error in REST Lookup for ${u}: ${err.message}`);
      if (conf.retryOnError) {
        retryInterval = Math.min(maxRetryInterval, retryInterval * 2);
        setTimeout(() => {
          dLogger.info(`Retrying for ${u} after ${retryInterval}`);
          lookupEvent(event, u, retryInterval, promise);
        }, retryInterval);
        return;
      }
      reject(`Error: ${err.message}`);
    };
    http.get(u, {
      agent
    }, (resp) => {
      if ((resp.statusCode < 200 || resp.statusCode > 299) && resp.statusCode != 404) {
        onError(new Error(`non-2XX status code: ${resp.statusCode}`));
        return;
      }
      let data = '';

      resp.on('data', (chunk) => {
        data += chunk;
      });

      resp.on('end', () => {
        let d = data;
        try {
          d = JSON.parse(data);
        } catch (e) {}
        if (conf.flatten && typeof d === 'object') {
          Object.keys(d).forEach(k => {
            event[`${conf.eventField}_${k}`] = d[k];
          });
        } else {
          event[conf.eventField] = d;
        }
        resolve(event);
      });

    }).on('error', onError)
  }));
}


exports.process = (event) => {
  const u = urlExpression.evalOn(event);

  // dLogger.info(`Executing REST Lookup against ${u}`);
  return lookupEvent(event, u, 1000);
};

