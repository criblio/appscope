import json
import collections
import datetime


class TestResult:
    def __init__(self, name, wrapped, std_out, std_err, duration, return_code):
        self.name = name
        self.wrapped = wrapped
        self.std_err = std_err
        self.std_out = std_out
        self.duration = duration
        self.return_code = return_code


class Watcher:

    passed_count = 0
    failed_count = 0
    start_date = None
    finish_date = None
    results = collections.defaultdict(dict)

    def start(self):
        self.start_date = datetime.datetime.now()

    def test_passed(self, result):
        self.passed_count += 1

        test_res = {
            "std_err" : result.std_err
        }

        if (result.wrapped):
            self.results[result.name]["wrapped"] = test_res
        else:
            self.results[result.name]["raw"] = test_res

    def test_failed(self, result):
        print(result.duration)
        self.failed_count += 1

    def finish(self):
        self.finish_date = datetime.datetime.now()
        print(json.dumps(self.results))

    def has_failures(self):
        return False
