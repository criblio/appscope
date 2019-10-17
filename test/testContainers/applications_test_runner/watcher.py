from datetime import datetime
from typing import List, Any

from common import TestSetResult, TestResult
from collections import defaultdict


class TestWatcher:

    def __init__(self, execution_id):
        self.execution_id = execution_id
        self.finish_date = None
        self.start_date = None
        self.__results = defaultdict(TestSetResult)
        self.__has_failures = False
        self.__error = None

    def test_execution_completed(self, name: str, duration: int, scoped: bool, result: TestResult, scope_messages: List[str], test_data: Any):
        test_results = self.__results[name]
        result_to_update = test_results.scoped_execution_data if scoped else test_results.unscoped_execution_data
        result_to_update.result = result
        result_to_update.duration = duration
        result_to_update.scope_messages = scope_messages
        result_to_update.test_data = test_data

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

    def finish_with_error(self, error):
        self.__has_failures = True
        self.__error = error
        self.finish_date = datetime.now()
