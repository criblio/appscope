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

        logging.info(f"Starting app {self.name} in {'scoped' if scoped else 'unscoped'} mode.")

        start_command = "/kafka/bin/zookeeper-server-start.sh /kafka/config/zookeeper.properties"

        logging.debug(f"Command is {start_command}.")
        subprocess.Popen(start_command.split(), env=env, start_new_session=True, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

        start_command = "/kafka/bin/kafka-server-start.sh /kafka/config/server.properties"

        logging.debug(f"Command is {start_command}.")
        subprocess.Popen(start_command.split(), env=env, start_new_session=True, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

        time.sleep(20)

    def stop(self):
        logging.info(f"Stopping app {self.name}.")
        subprocess.Popen("/kafka/bin/kafka-server-stop.sh", start_new_session=True)
        subprocess.Popen("/kafka/bin/zookeeper-server-stop.sh", start_new_session=True)
        time.sleep(20)

    def assert_running(self):
        completed_proc = subprocess.run("/kafka/bin/zookeeper-shell.sh localhost:2181 ls /brokers/ids", shell=True,
                                        universal_newlines=True, stdout=PIPE)
        assert "[0]" in completed_proc.stdout, "kafka is not running"


class KafkaMsgTest(ApplicationTest):

    def do_run(self, scoped) -> Tuple[TestResult, Any]:
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
