exports.disabled = 0;
exports.name = 'Cardinality Guard';
exports.version = 0.1;
exports.group = 'Demo Functions';

const { CardinalityGuard } = require('./cardinalityguard');
const NestedPropertyAccessor = C.expr.NestedPropertyAccessor;
const dLogger = C.util.getLogger('cardinalityGuard');

let conf;
let cg;

exports.init = (opt) => {
  conf = (opt || {}).conf || {};

  conf.buckets = conf.buckets || 10;
  conf.bucketSeconds = conf.bucketSeconds || 60;
  conf.maxValues = conf.maxValues || 1000;
  conf.ignoreFields = conf.ignoreFields || ['_*'];
  conf.redactValue = conf.redactValue || 'REDACTED';

  let WL22ignore;
  if (conf.ignoreFields.length > 0) {
    WL2ignore = new C.util.WildcardList(conf.ignoreFields);
  }

  cg = new CardinalityGuard(conf);
  exports.cg = cg;
};


exports.process = (event) => {
  const now = cg.getNow(event);
  NestedPropertyAccessor.traverseAndUpdate(event, 5, (path, value) => {
    if (!event.isInternalField(path) && !WL2ignore.test(path)) {
      let redact = cg.checkFieldValue(now, path, value);
      if (redact) {
        return conf.redactValue;
      }
    }
    return value;
  });
  return event;
};
