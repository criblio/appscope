import os
import subprocess
import time
import logging
from typing import Any, List

from common import AppController, TestResult
from runner import Runner
from validation import TestExecutionValidator, validate_all
from web import TestGetUrl, TestPostToUrl


class NginxApplicationController(AppController):
    name = "nginx"

    def __init__(self):
        self.proc = None

    def start(self, scoped=False):
        logging.info(f"Starting app {self.name} in {'scoped' if scoped else 'unscoped'} mode.")
        cmd = ["nginx", "-g", "daemon off;"]
        env = os.environ.copy()
        if scoped:
            env["LD_PRELOAD"] = "/usr/lib/libscope.so"
        logging.debug(f"Command is {cmd}. Environment {env}")

        self.proc = subprocess.Popen(cmd, env=env)
        time.sleep(5)
        assert self.is_running()

    def stop(self):
        self.proc.terminate()
        self.proc.communicate()

        time.sleep(3)
        assert not self.is_running()

    def is_running(self):
        return self.proc.poll() is None


class NetworkMetricsCollectedValidator(TestExecutionValidator):

    def __init__(self, applicable_tests: List[str]):
        self.applicable_tests = applicable_tests

    def should_validate(self, name: str, scoped: bool) -> bool:
        return scoped and name in self.applicable_tests

    def validate(self, test_data: Any, scope_messages: List[str]) -> TestResult:
        return validate_all(
            (any("net.tx" in msg for msg in scope_messages if "#proc:nginx" in msg), "No 'net.tx' metrics is collected"),
            (any("net.rx" in msg for msg in scope_messages if "#proc:nginx" in msg), "No 'net.rx' metrics is collected")
        )


def configure(runner: Runner):
    runner.set_app_controller(NginxApplicationController())
    get_home_page = TestGetUrl(url="http://localhost/", requests=10000)
    post_file = TestPostToUrl(url="http://localhost/log/", requests=10000, post_file="./post.json")
    runner.add_test_execution_validators([NetworkMetricsCollectedValidator([get_home_page.name, post_file.name])])
    runner.add_tests([get_home_page, post_file])
