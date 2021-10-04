#!/usr/bin/ruby

require "socket"
require "thread"
require "openssl"

host = ARGV[0]
port = Integer(ARGV[1])

socket = TCPSocket.new(host, port)
expectedCert = OpenSSL::X509::Certificate.new(File.open("cert.pem"))
ssl = OpenSSL::SSL::SSLSocket.new(socket)
ssl.sync_close = true
ssl.connect
if ssl.peer_cert.to_s != expectedCert.to_s
  stderrr.puts "Unexpected certificate"
  exit(1)
end

$stdout.puts "sending request"
request = "GET /hey/you HTTP/1.1\r\n" + 
          "Host: 127.0.0.1:8765\r\n" + 
          "User-Agent: me\r\n" + 
          "Accept: */*\r\n\r\n"
$stdout.puts request
ssl.puts request

lineIn = ssl.gets
lineIn = lineIn.chomp
$stdout.puts "received response"
$stdout.puts lineIn
