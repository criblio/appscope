const https = require('https')

const protocol = "TLSv1_2_method"

const options = {
    hostname: '127.0.0.1',
    port: 8000,
    method: 'GET',
    secureProtocol: protocol,
    rejectUnauthorized: false
}

https.request(options, res => {
  let body = ''
  res.on('data', data => body += data)
  res.on('end', () => {
    console.log('response data: ' + body)
  })
}).on('error', err => {
  console.warn(err)
}).end()
