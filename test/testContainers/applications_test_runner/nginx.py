import os
import subprocess
import time
import logging
from web import TestGetUrl




class NginxApplicationController:
    name = "nginx"

    def __init__(self):
        self.proc = None

    def start(self, scoped=False, scope_config_path=None):
        cmd = ["nginx", "-g", "daemon off;"]
        env = os.environ.copy()
        if scoped:
            env["LD_PRELOAD"] = "/usr/lib/libwrap.so"
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


def get_tests():
    return [TestGetUrl(url="http://localhost/", requests=10000)]
