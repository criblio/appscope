import glob
import json
import logging
import os
import socket
import subprocess
import time
from typing import Tuple, Any

from retrying import retry

from common import ApplicationTest, TestResult
from proc import SubprocessAppController
from runner import Runner
from utils import random_string
from validation import validate_all


class CriblTCPToFileTest(ApplicationTest):

    def do_run(self, scoped) -> Tuple[TestResult, Any]:
        # all configured in environment
        # TODO configure in cribl API
        # cribl tcp input localhost:10003
        # cribl file /tmp/CriblOut-*.json

        events_num = 1000

        host = "127.0.0.1"
        port = 10003

        logging.info(f"Waiting for {host}:{port} listener...")
        timeout = 90
        netstat = subprocess.check_output(f"netstat -an | grep -w {port} || true", shell=True)
        while b'LISTEN' not in netstat:
            netstat = subprocess.check_output(f"netstat -an | grep -w {port} || true", shell=True)
            time.sleep(1)
            timeout -= 1
            if timeout <= 0:
                logging.info(f"Giving up waiting for {host}:{port} listener")
                break

        logging.info(f"Sending {events_num} events to cribl tcp input at {host}:{port}")

        sent_messages = []

        for i in range(events_num):

            if i % (events_num / 5) == 0:
                logging.info(f"Sent {i} events out of {events_num}")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))

                message = {
                    "message": random_string(10),
                    "source": "splunk_test_runner",
                    "count": i
                }
                s.sendall(json.dumps(message).encode("utf-8"))
                sent_messages.append(message)
                # short delay, cause cribl is losing messages on a heavy load??
                time.sleep(0.005)

        logging.info("Finished sending events.")

        out_file_pattern = "/tmp/CriblOut-*.json"
        logging.info(f"Waiting for cribl to output to file {out_file_pattern}")

        result_file_paths = self.__wait_for_files(out_file_pattern)

        try:
            res = self.__validate_results(result_file_paths, sent_messages)
        finally:
            for path in result_file_paths:
                os.remove(path)

        return res, None

    def __validate_results(self, result_file_paths, sent_messages):
        def read_msg_from_json(msg):
            obj = json.loads(msg)

            return {
                "message": obj["message"],
                "source": obj["source"],
                "count": obj["count"],
            }

        received_messages = []
        for path in result_file_paths:
            logging.info(f"Reading file {path}")
            with open(path, "r") as f:
                lines = f.readlines()
                logging.debug("First lines from file:")
                logging.debug("".join(lines[:5]))
                received_messages.extend([read_msg_from_json(l) for l in lines])

        logging.info("Validating results")
        return validate_all(
            (
                len(received_messages) == len(sent_messages),
                f"Expected to see {len(sent_messages)} messages in out file, but found {len(received_messages)}"
            ),
            (
                sorted(sent_messages, key=lambda k: k["count"]) == sorted(received_messages, key=lambda k: k["count"]),
                f"Sent messages differs from results"
            )
        )

    @retry(stop_max_attempt_number=7, wait_fixed=20000)
    def __wait_for_files(self, path):
        matched_files = glob.glob(path)
        if len(matched_files) == 0:
            raise FileNotFoundError(path)

        return matched_files

    @property
    def name(self):
        return "tcp to file passthrough"


def configure(runner: Runner, config):
    app_controller = SubprocessAppController(["/opt/cribl/bin/cribl", "server"], "cribl", config.scope_path,
                                             config.logs_path)
    runner.add_tests([CriblTCPToFileTest(app_controller)])
