const https = require('https');
const fs = require('fs');
const constants = require('crypto').constants;
const hostname = '127.0.0.1';
const port = 8000;

const options = {
  key: fs.readFileSync('key.pem'),
  cert: fs.readFileSync('cert.pem'),
  requestCert: false,
  rejectUnauthorized: false,
  secureOptions: constants.SSL_OP_NO_SSLv2 | constants.SSL_OP_NO_SSLv3 | constants.SSL_OP_NO_TLSv1
};

https.createServer(options, function (req, res) {
  res.writeHead(200);
  res.end("hello world\n");
}).listen(port, hostname, () => {
  console.log(`Server running at https://${hostname}:${port}/`);
});
