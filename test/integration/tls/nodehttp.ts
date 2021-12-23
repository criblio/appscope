// var url = "https://www.google.com"
// var url = "https://wttr.in/Deadwood"
var url = "https://cribl.io"

const https = require('https');
const req = https.request(url, res => {
  console.log(`statusCode: ${res.statusCode}`)
  res.on('data', d => process.stdout.write(d));
});
req.on('error', error => console.error(error));
req.end();
