# file name is "kafka_test" to avoid conflict with library "kafka"
import logging
import os
import subprocess
import time
from subprocess import PIPE
from typing import Tuple, Any

from kafka import KafkaConsumer
from kafka import KafkaProducer

from common import ApplicationTest, AppController, TestResult
from runner import Runner
from validation import passed


class KafkaAppController(AppController):

    def __init__(self, scope_path):
        super().__init__("kafka")
        self.scope_path = scope_path
        self.proc = None

    def start(self, scoped):
        env = os.environ.copy()
        if scoped:
            env["LD_PRELOAD"] = self.scope_path

        logging.info("Sockets before start()")
        os.system('netstat -an | grep -w 9092')
        logging.info("ps before start()")
        os.system('ps -ef')

        logging.info(f"Starting app {self.name} in {'scoped' if scoped else 'unscoped'} mode.")

        start_command = "/kafka/bin/zookeeper-server-start.sh /kafka/config/zookeeper.properties"

        logging.debug(f"Command is {start_command}.")
        self.zookeper = subprocess.Popen(start_command.split(), env=env,
                start_new_session=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

        start_command = "/kafka/bin/kafka-server-start.sh /kafka/config/server.properties"

        logging.debug(f"Command is {start_command}.")
        self.server = subprocess.Popen(start_command.split(), env=env,
                start_new_session=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

        time.sleep(30)

        logging.info("Sockets after start()")
        os.system('netstat -an | grep -w 9092')
        logging.info("ps after start()")
        os.system('ps -ef')

    def stop(self):
        # https://kafka.apache.org/quickstart says CTRL-C to stop these and
        # these scropts are not working reliably so switching to sending
        # SIGTERM and wait().
        subprocess.Popen("/kafka/bin/kafka-server-stop.sh", start_new_session=True)
        subprocess.Popen("/kafka/bin/zookeeper-server-stop.sh", start_new_session=True)
        #self.server.terminate();   self.server.wait()
        #self.zookeper.terminate(); self.zookeper.wait()

        time.sleep(30)

        logging.info("Sockets after stop()")
        os.system('netstat -an | grep -w 9092')
        logging.info("ps after stop()")
        os.system('ps -ef')

    def assert_running(self):
        completed_proc = subprocess.run("/kafka/bin/zookeeper-shell.sh localhost:2181 ls /brokers/ids", shell=True,
                                        universal_newlines=True, stdout=PIPE)
        assert "[0]" in completed_proc.stdout, "kafka is not running"


class KafkaMsgTest(ApplicationTest):

    def do_run(self, scoped) -> Tuple[TestResult, Any]:
        logging.info("9092 Sockets before do_run()")
        os.system('netstat -an | grep -w 9092')
        logging.info("ps before do_run()")
        os.system('ps -ef')

        logging.info(f"Connecting to Kafka")
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

        msg_count = 100
        for _ in range(msg_count):
            producer.send('test-topic', b'msg')

        consumer = KafkaConsumer('test-topic', group_id='group', bootstrap_servers=['localhost:9092'],
                                 auto_offset_reset='earliest', enable_auto_commit=False, session_timeout_ms=10000,
                                 max_poll_interval_ms=10000)

        msg_count_recv = 0
        for _ in range(msg_count):
            msg = next(consumer)
            msg_count_recv += 1

        assert msg_count == msg_count_recv, f"Expected to have {msg_count}, but received {msg_count_recv}"
        return passed(), None

    @property
    def name(self):
        return "kafka"


def configure(runner: Runner, config):
    app_controller = KafkaAppController(config.scope_path)

    runner.add_tests([KafkaMsgTest(app_controller)])
