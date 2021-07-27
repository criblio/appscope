import json
import logging
import os
import subprocess
from typing import Tuple, Any, List

from common import Test, TestResult
from runner import Runner
from utils import dotdict
from validation import validate_all

config = dotdict({
    "timeout": 10000
})


class SyscallTest(Test):

    def __init__(self, path):
        self.path = path

    def run(self, scoped) -> Tuple[TestResult, Any]:
        env = os.environ.copy()
        if scoped:
            env["LD_PRELOAD"] = config.scope_path

        logging.debug(f"Command is {self.path}. Environment {env}")
        completed_proc = subprocess.run([self.path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, timeout=config.timeout)

        result = validate_all(
            (completed_proc.returncode == 0, f"Syscall test exited with return code {completed_proc.returncode}"))

        if not result.passed:
            logging.warning(completed_proc.stderr)

        data = {
            "return_code": completed_proc.returncode,
            "stdout": completed_proc.stdout,
            "stderr": completed_proc.stderr
        }

        return result, data

    @property
    def name(self):
        # filename without extension
        return os.path.splitext(os.path.basename(self.path))[0]


def locate_tests(root_path: str, included_tests: List[str], excluded_tests: List[str]) -> List[SyscallTest]:
    def locate_in_dir(dir_path: str, excluded_tests: List[str]) -> List[SyscallTest]:

        logging.debug(f"Visiting {dir_path}")
        tests = []

        items_to_visit = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i not in excluded_tests]
        logging.debug(f"Found  {len(items_to_visit)} items to visit.")

        for item in items_to_visit:
            if os.path.isdir(item):
                tests.extend(locate_in_dir(item, excluded_tests))
            elif os.path.isfile(item) and os.access(item, os.X_OK):
                tests.append(SyscallTest(path=item))
            else:
                logging.debug(f"{item} is not suitable for tests")

        return tests

    logging.debug(
        f"Locating tests. Root path: {root_path}, included tests: {included_tests}, excluded tests {excluded_tests}")

    if len(included_tests) > 0:
        locations = [os.path.join(root_path, i) for i in included_tests]
    else:
        locations = [root_path]

    tests = []
    for location in locations:
        tests.extend(locate_in_dir(location, excluded_tests))

    logging.info(f"Location {root_path}. {len(tests)} tests found.")
    return tests


def configure(runner: Runner, app_config):
    config_path = app_config.syscalls_tests_config
    config.scope_path = app_config.scope_path

    logging.info(f"Reading syscalls tests config from file {config_path}")
    with open(config_path, "r") as f:
        syscalls_tests_config = json.load(f)

    for location in syscalls_tests_config["locations"]:
        enabled = location.get("enabled", True)
        home_dir = location["home"]
        if not enabled:
            logging.debug(f"Location {home_dir} is ignored")
            continue

        included_tests = location.get("include_tests", [])
        excluded_tests = location.get("exclude_tests", [])
        arch = subprocess.check_output(["uname","-m"])
        if b'x86_64' in arch:
            excluded_tests = excluded_tests + location.get("exclude_x86_64_tests", [])
        elif b'aarch64' in arch:
            excluded_tests = excluded_tests + location.get("exclude_aarch_tests", [])

        tests = locate_tests(home_dir, included_tests, excluded_tests)

        runner.add_tests(tests)
