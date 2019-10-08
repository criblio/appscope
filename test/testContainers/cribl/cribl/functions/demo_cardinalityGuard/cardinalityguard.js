//////////// TESTING SETUP
// Get Cribl logger and NestedPropertyAccessor
// or override them for testing outside of Cribl
let dLogger;
let NestedPropertyAccessor;
if (process.env.NODE_ENV === 'TEST') {
  dLogger = {
    'info': (...logItems) => {
      console.log(...logItems);
    }
  }
  NestedPropertyAccessor = {
    'traverseAndUpdate': (event, depth, cb) => {
      // Implement flat traverse for testing
      event.keys().forEach(k => {
        event[k] = cb(k, event[k]);
      })
    }
  }
} else {
  dLogger = C.util.getLogger('cardinalityGuard');
  NestedPropertyAccessor = C.expr.NestedPropertyAccessor;
}
/////////// END TESTING SETUP

class CardinalityGuard {
  constructor(conf) {
    this.buckets = conf.buckets;
    this.bucketSeconds = conf.bucketSeconds;
    this.maxValues = conf.maxValues;
    this.totalTime = conf.buckets * conf.bucketSeconds * 1000;
    this.startTime = conf.startTime; // Used by unit tests
    this._entries = {};
    this._currentBucketIdx = 0;
  }

  // getBucketIdx retrieves the right Set from a list of sets conf.buckets long. 
  // Based on the timer since we've started, getBucket gets the amount of time we're
  // in this current interval, and then determines the bucket based on which bucket
  // we're in in this interval.
  getBucketIdx(now) {
    if (this.startTime === undefined || this.startTime > now) {
      this.startTime = now;
    }
    const timeSinceStart = now - this.startTime;
    const timeInInterval = timeSinceStart % this.totalTime;
    const ret = Math.floor(timeInInterval / (this.bucketSeconds * 1000));
    if (ret < 0) {
      throw new Error(`bucketIdx < 0, now: ${now} startTime: ${this.startTime} totalTime: ${this.totalTime}`)
    }
    return ret;
  }

  // getSetList retrieves the proper setList to check aginst. It checks the entries
  // map to get the right list of sets based on the field name. If missing, it creates
  // an empty set list.
  getSetList(field) {
    let setList = this._entries[field];
    if (setList === undefined) {
      setList = Array.apply(null, Array(this.buckets)).map((x, i) => {
        return new Set()
      });
      this._entries[field] = setList;
    }
    return setList;
  }

  // getNow looks for time in the event, if it does not find it, it uses current time
  getNow(event) {
    let ret;
    if (typeof (event) === 'object' && event._time) {
      ret = event._time * 1000;
    } else {
      ret = Date.now();
    }
    if (this.startTime === undefined || this.startTime > ret) {
      this.startTime = ret;
    }
    return ret;
  }

  // valueCount sums the set sizes in the setList
  valueCount(field) {
    return this.getSetList(field).reduce((prev, cur) => {
      return prev + cur.size
    }, 0);
  }

  // setFieldValue sets a value for a field in the SetList in the proper bucket
  setFieldValue(now, field, value) {
    const bucketIdx = this.getBucketIdx(now);
    const set = this.getSetList(field)[bucketIdx];
    if (!set) {
      throw new Error(`set undefined for now: ${now} field: ${field} value: ${value} bucketIdx: ${bucketIdx}`)
    }
    if (bucketIdx != this._currentBucketIdx) {
      set.clear();
      this._currentBucketIdx = bucketIdx;
    }
    if (set.size < this.maxValues) {
      set.add(value);
    }
    return set.size;
  }

  // checkFieldValue returns whether to redact a given field and value
  checkFieldValue(now, field, value) {
    const setSize = this.setFieldValue(now, field, value);
    if (setSize >= this.maxValues) {
      return true;
    }
    const valueCount = this.valueCount(field);
    if (valueCount >= this.maxValues) {
      return true;
    }
    return false;
  }
}

exports.CardinalityGuard = CardinalityGuard;
