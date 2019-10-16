from datetime import datetime

from common import TestSetResult
from collections import defaultdict


class TestWatcher:

    def __init__(self):
        self.finish_date = None
        self.start_date = None
        self.__results = defaultdict(TestSetResult)
        self.__has_failures = False
        self.__error = None

    def test_execution_completed(self, name, duration, scoped, result):
        result_to_update = (self.__results[name].unscoped_execution_data, self.__results[name].scoped_execution_data)[
            scoped]
        result_to_update.result = result
        result_to_update.duration = duration

    def test_validated(self, name, passed, error=None):
        if not passed:
            self.__has_failures = False
        self.__results[name].passed = passed
        self.__results[name].error = error

    def has_failures(self):
        return self.__has_failures

    def get_test_results(self, name):
        return self.__results[name]

    def get_all_results(self):
        return self.__results

    def start(self):
        self.start_date = datetime.now()

    def finish(self):
        self.finish_date = datetime.now()

    def finishWithError(self, error):
        self.__has_failures = True
        self.__error = error
        self.finish_date = datetime.now()
