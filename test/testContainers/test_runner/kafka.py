import logging
from logging import DEBUG
from typing import Tuple, Any
import subprocess
from subprocess import PIPE
import time

from common import ApplicationTest, AppController, TestResult
from proc import SubprocessAppController
from runner import Runner
from utils import dotdict, random_string
from validation import passed, validate_all

import sys
import imp
import os

fp, pathname, description = imp.find_module('kafka', [s for s in sys.path if "site-packages" in s])
kafka_python = imp.load_module('kafka', fp, pathname, description)

from kafka import KafkaProducer
from kafka import KafkaConsumer

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
        subprocess.Popen(start_command.split(), env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(5)
	
        start_command = "/kafka/bin/kafka-server-start.sh /kafka/config/server.properties"

        logging.debug(f"Command is {start_command}.")
        subprocess.Popen(start_command.split(), env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        time.sleep(10)

    def stop(self):
        subprocess.Popen("/kafka/bin/kafka-server-stop.sh", start_new_session=True)
        subprocess.Popen("/kafka/bin/zookeeper-server-stop.sh", start_new_session=True)
        time.sleep(5)

    def assert_running(self):
        completed_proc = subprocess.run("/kafka/bin/zookeeper-shell.sh localhost:2181 ls /brokers/ids", shell=True, universal_newlines=True, stdout=PIPE)
        assert "[0]" in completed_proc.stdout, "kafka is not running"

class KafkaMsgTest(ApplicationTest):

    def do_run(self, scoped) -> Tuple[TestResult, Any]:
        logging.info(f"Connecting to Kafka")
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
        
        msg_count = 100
        for _ in range(msg_count):
            producer.send('test-topic', b'msg')
    
        consumer = KafkaConsumer('test-topic', group_id='group', bootstrap_servers=['localhost:9092'], auto_offset_reset = 'earliest', enable_auto_commit=False, session_timeout_ms=10000, max_poll_interval_ms= 10000)

        msg_count_recv = 0
        for _ in range(msg_count):
            msg=next(consumer)
            msg_count_recv += 1
            #print("%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value))
    
        assert msg_count == msg_count_recv, f"Expected to have {msg_count}, but received {msg_count_recv}"
        return passed(), None
        
    @property
    def name(self):
        return "kafka"

def configure(runner: Runner, config):
    app_controller = KafkaAppController(config.scope_path)

    runner.add_tests([KafkaMsgTest(app_controller)])