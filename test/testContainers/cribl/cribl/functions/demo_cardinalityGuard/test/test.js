const { expect } = require('chai');
const { CardinalityGuard } = require('../cardinalityguard');

process.env.NODE_ENV = 'test';

now = Date.now();

let cg;

describe('cardinalityGuard', () => {
  before('run init', () => {
    cg = new CardinalityGuard({
      buckets: 3,
      bucketSeconds: 1,
      maxValues: 3,
      startTime: now
    });
  });
  it('should get the right now', () => {
    inEvent = cg.getNow({ _time: now/1000 });
    expect(inEvent).to.equal(now);
    nowNow = cg.getNow()
    expect(nowNow - now).to.be.greaterThan(0).and.lessThan(5000);
  });
  it('assign the right buckets based on time', () => {
    expect(cg.getBucketIdx(now+1560)).to.equal(1);
    expect(cg.getBucketIdx(now+0400)).to.equal(0);
    expect(cg.getBucketIdx(now+2211)).to.equal(2);
    expect(cg.getBucketIdx(now+3932)).to.equal(0);
    expect(cg.getBucketIdx(now+7612)).to.equal(1);
  });
  it('should retrieve an empty set list for a field with conf.buckets number of entries', () => {
    const setList = cg.getSetList('foo')
    expect(setList.length).to.equal(3);
    setList.forEach(s => {
      expect(s.size).to.equal(0);
    });
    delete cg._entries.foo;
  });
  it('should properly track max values for a bucket', () => {
    let ret = cg.setFieldValue(now+0100, 'myfoo', 'myvalue');
    expect(ret).to.equal(1);

    // Duplicates don't increase it
    ret = cg.setFieldValue(now + 0100, 'myfoo', 'myvalue');
    expect(ret).to.equal(1);

    ret = cg.setFieldValue(now + 0100, 'myfoo', 'myvalue2');
    expect(ret).to.equal(2);
    ret = cg.setFieldValue(now + 0100, 'myfoo', 'myvalue3');
    expect(ret).to.equal(3);

    // We cap at max values
    ret = cg.setFieldValue(now + 0100, 'myfoo', 'myvalue4');
    expect(ret).to.equal(3);

    // New bucket
    ret = cg.setFieldValue(now + 1100, 'myfoo', 'myvalue4');
    expect(ret).to.equal(1);
    expect(cg.valueCount('myfoo')).to.equal(4);

    // And Another New bucket
    ret = cg.setFieldValue(now + 2100, 'myfoo', 'myvalue4');
    expect(ret).to.equal(1);
    expect(cg.valueCount('myfoo')).to.equal(5);

    // And Another New bucket
    ret = cg.setFieldValue(now + 3100, 'myfoo', 'myvalue4');
    expect(ret).to.equal(1);
    // Old bucket should have rolled off
    expect(cg.valueCount('myfoo')).to.equal(3);
  });
  it('checkFieldValue should return proper values', () => {
    let ret = cg.checkFieldValue(now + 0100, 'myfoo2', 'myvalue');
    expect(ret).to.be.false;

    // Duplicates don't increase it
    ret = cg.checkFieldValue(now + 0100, 'myfoo2', 'myvalue');
    expect(ret).to.be.false;

    ret = cg.checkFieldValue(now + 0100, 'myfoo2', 'myvalue2');
    expect(ret).to.be.false;
    ret = cg.checkFieldValue(now + 0100, 'myfoo2', 'myvalue3');
    expect(ret).to.be.true;

    // We cap at max values
    ret = cg.checkFieldValue(now + 0100, 'myfoo2', 'myvalue4');
    expect(ret).to.be.true;

    // New bucket
    ret = cg.checkFieldValue(now + 1100, 'myfoo2', 'myvalue4');
    expect(ret).to.be.true;

    // And Another New bucket
    ret = cg.checkFieldValue(now + 2100, 'myfoo2', 'myvalue4');
    expect(ret).to.be.true;

    // And Another New bucket
    ret = cg.checkFieldValue(now + 3100, 'myfoo2', 'myvalue4');
    expect(ret).to.be.true;
  });
});