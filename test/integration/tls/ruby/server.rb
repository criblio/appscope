#!/usr/bin/ruby

require "socket"
require "openssl"
require "thread"

listeningPort = Integer(ARGV[0])

server = TCPServer.new(listeningPort)
sslContext = OpenSSL::SSL::SSLContext.new
sslContext.cert = OpenSSL::X509::Certificate.new(File.open("cert.pem"))
sslContext.key = OpenSSL::PKey::RSA.new(File.open("priv.pem"))
sslServer = OpenSSL::SSL::SSLServer.new(server, sslContext)

puts "Listening on port #{listeningPort}"

connection = sslServer.accept
request = ""
while (lineIn = connection.gets)
  lineIn = lineIn.chomp
  request = request + lineIn + "\n"
  break if lineIn.bytesize == 0
end
$stdout.puts "received request"
$stdout.puts request

$stdout.puts "sending response"
response = "HTTP/1.1 200 OK\r\n\r\n"
connection.puts response

sleep 1
