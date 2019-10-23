import logging
import os
import subprocess
import time
from subprocess import PIPE
from typing import Tuple, Any

from common import Test, TestResult, AppController
from runner import Runner
from validation import passed


class SplunkAppController(AppController):

    def __init__(self):
        super().__init__("splunk")
        self.proc = None

    def start(self, scoped):
        logging.info(f"Starting app {self.name} in {'scoped' if scoped else 'unscoped'} mode.")
        self.__update_launch_conf(scoped)
        start_command = "/opt/splunk/bin/splunk start --accept-license --answer-yes --seed-passwd cribldemo"

        logging.debug(f"Command is {start_command}.")
        subprocess.run(start_command, check=True, shell=True)

    def stop(self):
        subprocess.run("/opt/splunk/bin/splunk stop", check=True, shell=True)

    def is_running(self):
        completed_proc = subprocess.run("/opt/splunk/bin/splunk status", check=True, shell=True,
                                        universal_newlines=True, stdout=PIPE)
        return "splunkd is running" in completed_proc.stdout

    def __update_launch_conf(self, scoped):
        splunk_home = os.environ["SPLUNK_HOME"]
        # TODO move path to scope to config
        scope_env_var = "LD_PRELOAD=/usr/lib/libscope.so"
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


class SplunkDummyTest(Test):

    @property
    def name(self):
        return "dummy test"

    def run(self) -> Tuple[TestResult, Any]:
        time.sleep(2)
        return passed(), None


def configure(runner: Runner):
    runner.set_app_controller(SplunkAppController())
    runner.add_tests([SplunkDummyTest()])
