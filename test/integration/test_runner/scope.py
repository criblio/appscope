import abc
import logging
import socketserver
import subprocess
import time
import threading

class ScopeDataCollector:

    def __init__(self, metric_type: str) -> None:
        default_timeouts = {"udp": 1, "tcp": 0.05}

        self.timeout = default_timeouts[metric_type]
        self.__messages = []

    def add(self, msg: str) -> None:
        self.__messages.append(msg)

    def get_all(self) -> list:
        return self.__messages

    def reset(self) -> None:
        logging.debug("Reset collector.")
        self.__messages = []

    def wait(self) -> None:
        time.sleep(self.timeout)


class DataListener(abc.ABC):

    def __init__(self, collector: ScopeDataCollector, host: str, port: int) -> None:
        self.port = port
        self.host = host
        self.collector = collector

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def start_server(self) -> None:
        pass

    @abc.abstractmethod
    def stop_server(self) -> None:
        pass

    @abc.abstractmethod
    def create_server(self) -> None:
        pass

    def start_server_thread(self, server: socketserver.BaseServer) -> None:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()

    def __enter__(self) -> 'DataListener':
        self.start_server()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.stop_server()

class ScopeUDPDataListener(DataListener):
    __server = None
    class __Handler(socketserver.DatagramRequestHandler):
        collector = None

        def handle(self):
            self.collector.add(self.rfile.read().decode())

    @property
    def name(self) -> str:
        return "UDP"

    def create_server(self) -> socketserver.ThreadingUDPServer:
        self.__Handler.collector = self.collector
        logging.info(f"Starting scope {self.name} listener at {self.host}:{self.port}")
        return socketserver.ThreadingUDPServer((self.host, self.port), self.__Handler)

    def start_server(self) -> None:
        self.__server = self.create_server()  
        self.start_server_thread(self.__server)

    def stop_server(self) -> None:
         logging.info(f"Stopping scope {self.name} listener.")
         self.__server.shutdown()

class ScopeTCPDataListener(DataListener):
    server = None

    class __Handler(socketserver.StreamRequestHandler):
        collector = None

        def handle(self):
            for line in self.rfile.readlines():  
                self.collector.add(line.decode())

    @property
    def name(self) -> str:
        return "TCP"

    def create_server(self) ->  socketserver.ThreadingTCPServer:
        self.__Handler.collector = self.collector
        logging.info(f"Starting scope {self.name} listener at {self.host}:{self.port}")
        return socketserver.ThreadingTCPServer((self.host, self.port), self.__Handler)

    def start_server(self) -> None:
        self.__server = self.create_server()  
        self.start_server_thread(self.__server)

    def stop_server(self) -> None:
         logging.info(f"Stopping scope {self.name} listener.")
         self.__server.shutdown()

def create_listener(type: str, collector: ScopeDataCollector) -> DataListener:
    default_listeners = {"udp": ScopeUDPDataListener, "tcp": ScopeTCPDataListener}
    
    listener = default_listeners[type]
    return listener(collector, "localhost", 8125)

