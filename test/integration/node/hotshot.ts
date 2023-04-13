// From: https://github.com/brightcove/hot-shots

var StatsD = require('hot-shots'),
    client = new StatsD({
        port: 8125,
        globalTags: { source: "hot-shots" }
    });

console.log("Starting to generate metrics from hotshot.ts");

client.increment("my.counter");
client.increment("my.counter");
client.decrement("my.counter");
client.histogram("my.histogram", 42);
client.gauge("my.gauge", 66.6);
client.increment(["my.counter", "my.other_counter"]);

console.log("Completed metric generation from hotshot.ts");
