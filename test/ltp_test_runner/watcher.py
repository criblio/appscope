import json
import collections
import datetime
import os
import sys


class TestResult:
    def __init__(self, name, scoped, stdout, stderr, duration, return_code):
        self.name = name
        self.scoped = scoped
        self.stderr = stderr
        self.stdout = stdout
        self.duration = duration
        self.return_code = return_code


class Watcher:

    def __init__(self, log_path):
        self._log_path = log_path
        self._start_date = None
        self._finish_date = None
        self._results = collections.defaultdict(dict)
        self._has_failures = False
        self._log_file = None
        self._failed_tests_file = None

    def start(self):
        self._start_date = datetime.datetime.now()
        self._print("Started tests execution")

        log_file_path = os.path.join(self._log_path, "log_" + self._start_date.isoformat() + ".json")
        failed_tests_file_path = os.path.join(self._log_path, "log_" + self._start_date.isoformat() + ".json")
        self._log_file = open(log_file_path, "a")
        self._failed_tests_file = open(failed_tests_file_path, "a")
        self._print("Logs can be found in " + log_file_path)

    def test_passed(self, result):
        self._test_completed(result)

    def test_failed(self, result):
        self._has_failures = True
        self._test_completed(result)

    def finish(self):
        self._finish_date = datetime.datetime.now()
        self._print("Finished tests execution")
        if self._has_failures:
            self._print("**** TEST FAILURES DETECTED ****")
            self._print("See " + self._failed_tests_file.name)
        self._print_summary()
        self._log_summary()

        self._log_file.close()
        self._failed_tests_file.close()

    def has_failures(self):
        return self._has_failures

    # -- private --
    def _test_completed(self, result):

        if result.scoped:
            self._results[result.name]["scoped"] = result
        else:
            self._results[result.name]["raw"] = result

        if self._results[result.name].get("raw") and self._results[result.name].get("scoped"):
            self._full_set_completed(self._results[result.name]["raw"], self._results[result.name]["scoped"])

    def _print(self, st):
        print(st)

    def _print_summary(self):

        def _to_test_result(code):
            return ("fail", "pass")[code == 0]

        self._print("Test results:")
        self._print("")
        self._print("Test\t\tRaw result\tWrapped result\tRaw duration\tWrapped duration")
        for k, v in self._results.items():
            self._print("{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}".format(
                k,
                _to_test_result(v["raw"].return_code),
                _to_test_result(v["scoped"].return_code),
                v["raw"].duration,
                v["scoped"].duration)
            )

    def _full_set_completed(self, raw_result, scoped_result):
        is_valid = self._validate(raw_result, scoped_result)
        self._results[raw_result.name]["failed"] = not is_valid
        log_row = self._to_log_row(raw_result, scoped_result, is_valid)
        self._log_file.write(log_row + "\n")
        if not is_valid:
            self._has_failures = True
            self._failed_tests_file.write(log_row + "\n")

    def _validate(self, raw_result, scoped_result):
        return raw_result.return_code == 0 and scoped_result.return_code == 0

    def _to_log_row(self, raw_result, scoped_result, is_valid):
        log_row = {
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

    def _log_summary(self):
        summary_row = json.dumps({
            "start_date": self._start_date.isoformat(),
            "finish_date": self._finish_date.isoformat(),
            "total_tests": len(self._results),
            "failed_tests": len(filter(lambda v: v["failed"], self._results.values())),
        })

        self._log_file.write(summary_row + "\n")
