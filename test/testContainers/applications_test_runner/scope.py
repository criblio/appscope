import logging
import socketserver
import subprocess
import threading


def get_scope_version(scope_path):
    completed_proc = subprocess.run([scope_path], universal_newlines=True, stdout=subprocess.PIPE)
    stdout = completed_proc.stdout
    for line in stdout.splitlines():
        if "Scope Version: " in line:
            return line.strip().replace("Scope Version: ", "")


class ScopeDataCollector:

    def __init__(self):
        self.__messages = []

    def add(self, msg):
        self.__messages.append(msg)

    def get_all(self):
        return self.__messages

    def reset(self):
        logging.debug("Reset collector.")
        self.__messages = []


class ScopeUDPDataListener:
    class __Handler(socketserver.DatagramRequestHandler):
        collector = None

        def handle(self):
            self.collector.add(self.rfile.read().decode())

    def __init__(self, collector: ScopeDataCollector, host, port):
        self.port = port
        self.host = host
        self.collector = collector
        self.__server = None

    def start(self):
        self.__Handler.collector = self.collector

        logging.info(f"Starting scope UDP listener at {self.host}:{self.port}")
        self.__server = socketserver.ThreadingUDPServer((self.host, self.port), self.__Handler)

        server_thread = threading.Thread(target=self.__server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()

    def stop(self):
        logging.info("Stopping scope UDP listener.")
        self.__server.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
