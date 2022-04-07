import json
import logging
import os
import subprocess
from subprocess import PIPE
from typing import Tuple, Any

import splunklib.client as client
from retrying import retry
from splunklib import results

from common import TestResult, AppController, ApplicationTest
from runner import Runner
from utils import random_string, dotdict
from validation import passed, validate_all

config = dotdict({
    "username": "admin",
    "password": "cribldemo"
})


class SplunkAppController(AppController):

    def __init__(self, scope_path):
        super().__init__("splunk")
        self.scope_path = scope_path
        self.proc = None

    def start(self, scoped):
        logging.info(f"Starting app {self.name} in {'scoped' if scoped else 'unscoped'} mode.")
        self.__update_launch_conf(scoped)
        start_command = "/opt/splunk/bin/splunk start --accept-license --answer-yes --seed-passwd cribldemo"

        logging.debug(f"Command is {start_command}.")
        subprocess.run(start_command, check=True, shell=True)

    def stop(self):
        subprocess.run("timeout 30 /opt/splunk/bin/splunk stop --force", check=False, shell=True)

    def assert_running(self):
        completed_proc = subprocess.run("/opt/splunk/bin/splunk status", check=True, shell=True,
                                        universal_newlines=True, stdout=PIPE)
        assert "splunkd is running" in completed_proc.stdout, "Splunkd is not running"

        service = client.connect(username=config.username, password=config.password)
        messages = client._load_atom_entries(service.get("messages"))
        logging.debug(messages)
        err_messages = [m['title'] for m in messages if m['content']['severity'] == 'error']
        assert len(err_messages) == 0, f"Error messages found {err_messages}"

    def __update_launch_conf(self, scoped):
        splunk_home = os.environ["SPLUNK_HOME"]
        scope_env_var = f"LD_PRELOAD={self.scope_path}"
        path_to_conf = os.path.join(splunk_home, "etc/splunk-launch.conf")
        logging.debug(
            f"Modifying splunk launch conf at {path_to_conf}. {'adding' if scoped else 'removing'} {scope_env_var}")

        with open(path_to_conf, "r") as f:
            lines = f.readlines()

        with open(path_to_conf, "w") as f:
            for line in lines:
                if scope_env_var not in line:
                    f.write(line)

            if scoped:
                f.write(f"\n{scope_env_var}\n")


class SplunkDirectIndexingTest(ApplicationTest):

    @property
    def name(self):
        return "splunk direct indexing"

    def do_run(self, scoped) -> Tuple[TestResult, Any]:
        logging.info(f"Connecting to Splunk. User {config.username}, Password {config.password}")

        service = client.connect(username=config.username, password=config.password)

        index_name = f"test_{random_string(4)}"
        logging.info(f"Creating test index '{index_name}'")
        index = service.indexes.create(index_name)

        events_num = 100000
        logging.info(f"Sending {events_num} events to Splunk")

        with index.attach() as sock:
            sock.settimeout(60)
            for i in range(events_num):
                sock.send(f"This is my test event {i}! \n".encode('utf-8'))

        self.__assert_same_number_of_events(service, events_num, index_name)
        self.__assert_transposed_results(service, index_name)

        return passed(), None

    @retry(stop_max_attempt_number=7, wait_fixed=8000)
    def __assert_same_number_of_events(self, service, expected_events_num, index_name):
        search_query = f"search index={index_name} | stats count"

        logging.info(f"Searching splunk for events. Query: {search_query}")
        search_results = service.jobs.oneshot(search_query)

        reader = results.ResultsReader(search_results)
        actual_events_num = reader.next()['count']
        logging.info(f"Found {actual_events_num} events in index {index_name}")

        assert expected_events_num == int(
            actual_events_num), f"Expected to see {expected_events_num} in index, but got only {actual_events_num}"

    def __assert_transposed_results(self, service, index_name):
        search_results = service.jobs.oneshot(f"search index={index_name} | head 10 | transpose")

        reader = results.ResultsReader(search_results)
        for result in reader:
            assert result.get("column") is not None, f"Attribute 'column' not found in a row {result}"


class SplunkKVStoreTest(ApplicationTest):

    def do_run(self, scoped) -> Tuple[TestResult, Any]:
        logging.info(f"Connecting to Splunk. User {config.username}, Password {config.password}")

        service = client.connect(username=config.username, password=config.password)
        service.namespace['owner'] = 'Nobody'

        collection_name = f"coll_{random_string(4)}"
        self.__create_collection(collection_name, service)

        collection = service.kvstore[collection_name]

        records_num = 100

        records = [json.dumps({"val": random_string(5)}) for i in range(records_num)]
        logging.info(f"Inserting {len(records)} records in kvstore")
        for record in records:
            collection.data.insert(record)

        found_records = collection.data.query()
        logging.info(f"Found {len(found_records)} records")
        return validate_all(
            (len(found_records) == records_num,
             f"Expected to have {records_num} records in kvstore, but found {len(found_records)}")
        ), None

    @retry(stop_max_attempt_number=7, wait_fixed=8000)
    def __create_collection(self, collection_name, service):
        collection = service.kvstore.create(collection_name)
        return collection

    @property
    def name(self):
        return "kvstore test"


def configure(runner: Runner, config):
    app_controller = SplunkAppController(config.scope_path)
    runner.add_tests([SplunkDirectIndexingTest(app_controller), SplunkKVStoreTest(app_controller)])
