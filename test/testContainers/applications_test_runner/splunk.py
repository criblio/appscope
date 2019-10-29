import logging
import os
import subprocess
from subprocess import PIPE
from typing import Tuple, Any

import splunklib.client as client
from retrying import retry
from splunklib import results

from common import Test, TestResult, AppController
from runner import Runner
from utils import random_string, dotdict
from validation import passed

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
        subprocess.run("/opt/splunk/bin/splunk stop", check=True, shell=True)

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


class SplunkDirectIndexingTest(Test):

    @property
    def name(self):
        return "splunk direct indexing"

    def run(self) -> Tuple[TestResult, Any]:
        logging.info(f"Connecting to Splunk. User {config.username}, Password {config.password}")

        service = client.connect(username=config.username, password=config.password)

        index_name = f"test_{random_string(4)}"
        logging.info(f"Creating test index '{index_name}'")
        index = service.indexes.create(index_name)

        events_num = 1000000
        logging.info(f"Sending {events_num} events to Splunk")

        with index.attach() as sock:
            sock.settimeout(60)
            for i in range(events_num):
                sock.send(f"This is my test event {i}! \n".encode('utf-8'))

        self.__assert_same_number_of_events(service, events_num, index_name)

        return passed(), None

    @retry(stop_max_attempt_number=7, wait_fixed=2000)
    def __assert_same_number_of_events(self, service, expected_events_num, index_name):
        search_query = f"search index={index_name} | stats count"

        logging.info(f"Searching splunk for events. Query: {search_query}")
        search_results = service.jobs.oneshot(search_query)

        # Get the results and display them using the ResultsReader
        reader = results.ResultsReader(search_results)
        actual_events_num = reader.next()['count']
        logging.info(f"Found {actual_events_num} events in index {index_name}")

        assert expected_events_num == int(
            actual_events_num), f"Expected to see {expected_events_num} in index, but got only {actual_events_num}"


def configure(runner: Runner, config):
    runner.set_app_controller(SplunkAppController(config.scope_path))
    runner.add_tests([SplunkDirectIndexingTest()])
