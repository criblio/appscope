import json


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

    def test_passed(self, result):
        print(result.duration)
        self.passed_count += 1

    def test_failed(self, result):
        print(result.duration)
        self.failed_count += 1
