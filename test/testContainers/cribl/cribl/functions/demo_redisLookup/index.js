exports.disabled = 0;
exports.name = 'Redis Lookup';
exports.version = 0.1;
exports.group = 'Demo Functions';

let conf;

const redis = require("redis");

let client;
let ready;

const dLogger = C.util.getLogger('redisLookup');

const redisConnect = () => {
  return new Promise(resolve => {
    dLogger.info(`connecting to redis ${conf.host}:${conf.port}`);
    client = redis.createClient(conf.port, conf.host, {
      tls: conf.secure ? { port: conf.port, host: conf.host } : null,
      db: conf.db,
    });
    client.on('connect', () => {
      dLogger.info(`connected to redis ${conf.host}:${conf.port}`);
      resolve();
    });
  });
}


exports.init = (opt) => {
  conf = (opt || {}).conf || {};

  ready = redisConnect();
};

exports.unload = () => {
  client.quit();
}


exports.process = (event) => {
  if (event[conf.fromField]) {
    return ready.then(() => new Promise((resolve, reject) => {
      client.get(`session_${event[conf.fromField]}`, (err, reply) => {
        if (err) reject(`Error: ${err.message}`);
        event[conf.toField] = reply;
        resolve(event);
      });
    }));
  }
  return event;
};

