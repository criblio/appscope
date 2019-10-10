import json
import collections
import datetime
import os
from utils import tabulate


class TestResult:
    def __init__(self, name, scoped, stdout, stderr, duration, return_code):
        self.name = name
        self.scoped = scoped
        self.stderr = stderr
        self.stdout = stdout
        self.duration = duration
        self.return_code = return_code


class Watcher:

    def __init__(self, log_path, execution_id, verbose):
        self.__verbose = verbose
        self.__execution_id = execution_id
        self.__log_path = log_path
        self.__start_date = None
        self.__finish_date = None
        self.__results = collections.defaultdict(dict)
        self.__has_failures = False
        self.__log_file = None
        self.__failed_tests_file = None

    def start(self):
        self.__start_date = datetime.datetime.now()
        self.__print("Started tests execution. Execution ID: " + self.__execution_id)
        if self.__verbose: self.__print("Verbose output is selected")

        self.__prepare_log_dir()
        log_file_path = os.path.join(self.__log_path, "log_" + self.__execution_id + ".json")
        failed_tests_file_path = os.path.join(self.__log_path, "failed_tests_" + self.__execution_id + ".json")
        self.__log_file = open(log_file_path, "a")
        self.__failed_tests_file = open(failed_tests_file_path, "a")
        self.__print("Logs can be found in " + log_file_path)

    def test_passed(self, result):
        self.__test_completed(result)

    def test_failed(self, result):
        self.__test_completed(result)

    def finish(self):
        self.__finish_date = datetime.datetime.now()
        self.__print("Finished tests execution")
        if self.__has_failures:
            self.__print("**** TEST FAILURES DETECTED ****")
            self.__print("See " + self.__failed_tests_file.name)
        self.__print_summary()
        self.__log_summary()

        self.__log_file.close()
        self.__failed_tests_file.close()

    def has_failures(self):
        return self.__has_failures

    # -- private --
    def __test_completed(self, result):

        if result.scoped:
            self.__results[result.name]["scoped"] = result
        else:
            self.__results[result.name]["raw"] = result

        if self.__results[result.name].get("raw") and self.__results[result.name].get("scoped"):
            self.__full_set_completed(self.__results[result.name]["raw"], self.__results[result.name]["scoped"])

    def __print(self, st):
        print(st)

    def __print_summary(self):

        def _to_test_result(code):
            return ("fail", "pass")[code == 0]

        cols_width = [20, 15, 15, 20, 20]
        self.__print("Test results:")
        self.__print("")
        self.__print(
            tabulate(["Test", "Raw result", "Scoped result", "Raw duration (ms)", "Scoped duration (ms)"], cols_width))
        for k, v in self.__results.items():
            self.__print(tabulate(
                [
                    k,
                    _to_test_result(v["raw"].return_code),
                    _to_test_result(v["scoped"].return_code),
                    v["raw"].duration,
                    v["scoped"].duration
                ],
                cols_width
            ))

    def __full_set_completed(self, raw_result, scoped_result):
        is_valid = self.__validate(raw_result, scoped_result)
        self.__results[raw_result.name]["failed"] = not is_valid
        log_row = self.__to_log_row(raw_result, scoped_result, is_valid)
        if self.__verbose: self.__print(log_row)
        self.__log_file.write(log_row + "\n")
        if not is_valid:
            self.__has_failures = True
            self.__failed_tests_file.write(log_row + "\n")

    def __validate(self, raw_result, scoped_result):
        return raw_result.return_code == scoped_result.return_code

    def __to_log_row(self, raw_result, scoped_result, is_valid):
        log_row = {
            "id": self.__execution_id,
            "test": raw_result.name,
            "status": ("fail", "pass")[is_valid],
            "raw": {
                "return_code": raw_result.return_code,
                "duration_ms": raw_result.duration,
                "stdout": raw_result.stdout,
                "stderr": raw_result.stderr
            },
            "scoped": {
                "return_code": scoped_result.return_code,
                "duration_ms": scoped_result.duration,
                "stdout": scoped_result.stdout,
                "stderr": scoped_result.stderr
            }
        }

        return json.dumps(log_row)

    def __log_summary(self):
        summary_row = json.dumps({
            "id": self.__execution_id,
            "status": ("fail", "pass")[self.__has_failures],
            "start_date": self.__start_date.isoformat(),
            "finish_date": self.__finish_date.isoformat(),
            "total_tests": len(self.__results),
            "failed_tests": len(filter(lambda v: v["failed"], self.__results.values())),
        })

        if self.__verbose: self.__print(summary_row)
        self.__log_file.write(summary_row + "\n")

    def __prepare_log_dir(self):
        if not os.path.exists(self.__log_path):
            self.__print("Log directory {0} not exists. Creating.".format(self.__log_path))
            os.makedirs(self.__log_path)
