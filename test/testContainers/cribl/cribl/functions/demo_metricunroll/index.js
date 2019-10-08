
exports.disabled = 0;
exports.name = 'Metric Unroll';
exports.version = '0.1';
exports.group = 'Demo Functions';

const INCLUDE = 0;

let conf = {};
let dimRex;
let valRex;

const cLogger = C.util.getLogger('func:metric_unroll');

function escapeRegExp(text) {
  return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&');
}

function wildcardStrToRegexPattern(str, anchor) {
  let regex = str;
  if (str.indexOf('*') > -1) {
    regex = `(?:${escapeRegExp(str).replace(/\\\*/g, '.*?')})`;
  }
  if (anchor) {
    regex = `^${regex}$`;
  }
  return regex;
}

exports.init = (opt) => {
  conf = (opt || {}).conf || {};

  conf.includeOrExclude = conf.includeOrExclude || INCLUDE;
  if (conf.includeOrExclude === INCLUDE) {
    conf.dimFields = conf.dimFields || ['*'];
    conf.valFields = conf.valFields || ['*'];
  } else {
    conf.dimFields = conf.dimFields || [];
    conf.valFields = conf.valFields || [];
    dimRex = /^$/; // We want this to match nothing by default
    valRex = /^$/; // We want this to match nothing by default
  }

  if (conf.dimFields.length > 0) {
    dimRex = new RegExp(conf.dimFields.map(df => wildcardStrToRegexPattern(df, true)).join('|'));
  }
  if (conf.valFields.length > 0) {
    valRex = new RegExp(conf.valFields.map(vf => wildcardStrToRegexPattern(vf, true)).join('|'));
  }
};

exports.process = (event) => {
  // Create a list of fields which will become exploded events and fields which are dimensions
  const valFields = [];
  const metricsModel = event.clone();
  const incl = conf.includeOrExclude === INCLUDE;
  Object.keys(metricsModel).forEach(k => {
    let valField = false;
    // If value is a number, see if we should include as a value field
    if (typeof event[k] === 'number' && k !== '_time' && !k.startsWith('__')) {
      const match = valRex.test(k);
      if (incl === Boolean(match)) {
        valFields.push(k);
        valField = true;
      }
      metricsModel[k] = undefined;
    }

    // Delete unused dimensions
    if (!valField) {
      const match = dimRex.test(k);
      if (!(incl === Boolean(match))) {
        metricsModel[k] = undefined;
      }
    }
  });

  return valFields.map(vf => {
    const newEvent = metricsModel.clone();
    newEvent._value = event[vf];
    newEvent.metric_name = vf;
    return newEvent;
  });
};

