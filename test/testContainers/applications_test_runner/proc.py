import logging
import os
import subprocess
import time

from common import AppController


class SubprocessAppController(AppController):

    def __init__(self, start_command, name, scope_path, start_wait=5, stop_wait=3):
        super().__init__(name)
        self.scope_path = scope_path
        self.stop_wait = stop_wait
        self.start_wait = start_wait
        self.start_command = start_command
        self.proc = None

    def start(self, scoped=False):
        logging.info(f"Starting app {self.name} in {'scoped' if scoped else 'unscoped'} mode.")
        env = os.environ.copy()
        if scoped:
            env["LD_PRELOAD"] = self.scope_path
        logging.debug(f"Command is {self.start_command}. Environment {env}")

        self.proc = subprocess.Popen(self.start_command, env=env)
        time.sleep(self.start_wait)
        self.assert_running()

    def stop(self):
        self.proc.terminate()
        self.proc.communicate()

        time.sleep(self.stop_wait)

    def assert_running(self):
        subprocess.run("ps -ef", shell=True)
        assert self.proc.poll() is None, f"{self.name} is crashed"


def wait_for_stdout_msg(proc, msg, timeout=60):
    timeout_start = time.time()

    while time.time() < timeout_start + timeout:
        line = proc.stdout.readline()
        if not line:
            break
        logging.debug(line)
        if msg in line:
            return

    raise TimeoutError(f"Message '{msg}' wasn't found in stdout after {timeout} seconds.")
