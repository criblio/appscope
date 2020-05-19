#!//usr/bin/python3

# To monitor traffic (confirm encryption over the wire)
#     sudo tcpdump -A -i lo port 8765
# As an alternate to using this script with the run_client argument
#     curl -v -k --cacert ./certs/selfsigned.crt --key ./certs/private.key https://127.0.0.1:8765/hey/you
#
# To run this script
#     test/manual/testssl.py create_certs
#     ./tcpserver 9109
#     LD_PRELOAD=lib/linux/libscope.so SCOPE_EVENT_HTTP=true test/manual/testssl.py start_server
#     LD_PRELOAD=lib/linux/libscope.so SCOPE_EVENT_HTTP=true test/manual/testssl.py run_client

import os
import socket
import ssl


CERT_DIR = './certs'
CERT_FILE = CERT_DIR + '/selfsigned.crt'
KEY_FILE = CERT_DIR + '/private.key'
PORT = 8765

def run_main():
    print (("Running {}...").format(script))
    if arg1 == "help":
        print_help()
    elif arg1 == "create_certs":
        create_certs()
    elif arg1 == "delete_certs":
        delete_certs()
    elif arg1 == "start_server":
        start_server()
    elif arg1 == "run_client":
        run_client()
    else:
        print('{} exiting with unknown argument {}...\n'.format(script, arg1))
        print_help()
    exit('{} exiting successfully.'.format(script))

def create_certs():
    print ("create_certs")
    os.system('mkdir -p {}'.format(CERT_DIR))

    # https://stackoverflow.com/questions/27164354/create-a-self-signed-x509-certificate-in-python
    from OpenSSL import crypto, SSL
    from socket import gethostname
    from pprint import pprint
    from time import gmtime, mktime

    # create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 1024)

    # create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().ST = "CA"
    cert.get_subject().L = "SanFrancisco"
    cert.get_subject().O = "CRIBL"
    cert.get_subject().OU = "CRIBL"
    cert.get_subject().CN = gethostname()
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10*365*24*60*60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha1')

    with open(CERT_FILE, "wt") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))
    with open(KEY_FILE, "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))

def delete_certs():
    print ("delete_certs")
    os.system('rm -rf {}'.format(CERT_DIR))

def start_server():
    print ("start_server")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
        sock.bind(('0.0.0.0', PORT))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.listen(5)
        with context.wrap_socket(sock, server_side=True) as ssock:
            conn, addr = ssock.accept()
            print("trying to receive something!")
            data = conn.recv(1024).decode('utf-8')
            print("received {}".format(data))
            conn.send("HTTP/1.1 200 OK\r\n\r\n".encode('utf-8'))
            print("sent reply")
            conn.close()


def run_client():
    print ("run_client")
    # PROTOCOL_TLS_CLIENT requires valid cert chain and hostname
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(CERT_FILE)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
        with context.wrap_socket(sock, server_hostname=socket.gethostname()) as ssock:
            print("trying to send something!")
            ssock.connect(('127.0.0.1', PORT))
            ssock.send("GET /hey/you HTTP/1.1\r\nHost: 127.0.0.1:8765\r\nUser-Agent: me\r\nAccept: */*\r\n\r\n".encode('utf-8'));
            rx_bytes = ssock.recv(4096)
            print("received: " + rx_bytes.decode('utf-8'))
            ssock.close()


def print_help():
    print ("  Legal operations:")
    print ("    {} create_certs".format(script))
    print ("    {} delete_certs".format(script))
    print ("    {} start_server".format(script))
    print ("    {} run_client".format(script))



from sys import argv
try:
    script, arg1 = argv
except:
    script = argv[0]
    if len(argv) == 1:
        print('{} exiting with missing argument...\n'.format(script))
    else:
        print('{} exiting with too many arguments...\n'.format(script))
    print_help()
    exit()

run_main()
