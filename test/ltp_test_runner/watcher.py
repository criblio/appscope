import json
import collections
import datetime
import sys


class TestResult:
    def __init__(self, name, wrapped, stdout, stderr, duration, return_code):
        self.name = name
        self.wrapped = wrapped
        self.stderr = stderr
        self.stdout = stdout
        self.duration = duration
        self.return_code = return_code


class Watcher:

    def __init__(self):
        self._start_date = None
        self._finish_date = None
        self._results = collections.defaultdict(dict)
        self._has_failures = False

    def start(self):
        self._start_date = datetime.datetime.now()
        self._print("Started tests execution")

    def test_passed(self, result):
        self._record_result(result)

    def test_failed(self, result):
        self._has_failures = True
        self._record_result(result)

    def finish(self):
        self._finish_date = datetime.datetime.now()
        self._print("Finished tests execution")
        if self._has_failures:
            self._print("**** TEST FAILURES DETECTED ****")
        self._print_summary()

    def has_failures(self):
        return self._has_failures

    # -- private --
    def _record_result(self, result):
        test_res = {
            "stderr" : result.stderr,
            "stdout" : result.stdout,
            "return_code" : result.return_code,
            "duration" : result.duration
        }

        if result.wrapped:
            self._results[result.name]["wrapped"] = test_res
        else:
            self._results[result.name]["raw"] = test_res

    def _print(self, st):
        print(st)

    def _print_summary(self):

        def _to_test_result(code):
            return ("FAIL", "pass")[code == 0]

        self._print("Test results:")
        self._print("")
        self._print("Test\t\tRaw result\tWrapped result\tRaw duration\tWrapped duration")
        for k,v in self._results.items():
            self._print("{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}".format(k,_to_test_result(v["raw"]["return_code"]), _to_test_result(v["wrapped"]["return_code"]), v["raw"]["duration"], v["wrapped"]["duration"]))





