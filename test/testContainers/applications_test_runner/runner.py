import logging
from datetime import datetime
from typing import List

from common import TestResult, Test
from scope import ScopeDataCollector
from utils import ms
from validation import TestExecutionValidator, TestSetValidator, passed
from watcher import TestWatcher


class Runner:
    __watcher: TestWatcher
    __tests: List[Test]
    __collector: ScopeDataCollector
    __test_execution_validators: List[TestExecutionValidator]
    __test_set_validators: List[TestSetValidator]

    def __init__(self, watcher, collector):
        self.__collector = collector
        self.__tests = []
        self.__watcher = watcher
        self.__test_execution_validators = []
        self.__test_set_validators = []

    def add_tests(self, tests: List[Test]):
        self.__tests.extend(tests)

    def add_test_execution_validators(self, test_execution_validators: List[TestExecutionValidator]):
        self.__test_execution_validators.extend(test_execution_validators)

    def add_test_set_validators(self, test_set_validators: List[TestSetValidator]):
        self.__test_set_validators.extend(test_set_validators)

    def run(self):

        logging.info(f"Tests: {[t.name for t in self.__tests]}")
        logging.info(f"Test execution validators: {[v.name for v in self.__test_execution_validators]}")
        logging.info(f"Test set validators: {[v.name for v in self.__test_set_validators]}")

        logging.info("Starting unscoped tests.")
        self.__execute_tests(False)
        logging.info("Unscoped tests execution finished.")

        logging.info("Starting scoped tests.")
        self.__execute_tests(True)
        logging.info("Scoped tests execution finished.")

        logging.info("Validating results.")
        self.__validate_results()

    def __validate_results(self):
        for test in self.__tests:
            result = self.__watcher.get_test_results(test.name)
            validation_res = self.__validate_test_set(test.name, result)
            self.__watcher.test_validated(test.name, validation_res.passed, validation_res.error)

    def __validate_test_set(self, name, result):
        for validator in self.__test_set_validators:
            if validator.should_validate(name):
                logging.debug(f"Test set {name}. Applying validator {validator.name}")
                res = validator.validate(result)
                if not res.passed:
                    logging.warning(f"Test set validator {validator.name} failed on {name}. Error: {res.error}")
                    return res

        return passed()

    def __validate_test_execution(self, name, scoped, test_data, scope_messages):
        for validator in self.__test_execution_validators:
            if validator.should_validate(name, scoped):
                logging.debug(f"Test execution {name}. Applying validator {validator.name}")
                res = validator.validate(test_data, scope_messages)
                if not res.passed:
                    logging.warning(f"Test execution validator {validator.name} failed on {name}. Error: {res.error}")
                    return res

        return passed()

    def __execute_tests(self, scoped):
        for test in self.__tests:

            logging.info("Running test %s", test.name)
            tick = datetime.now()
            result = None
            data = None
            try:
                result, data = test.run(scoped)
            except Exception as e:
                logging.warning(f"Test {test.name} failed with exception", exc_info=True)
                result = TestResult(passed=False, error="Exception: " + str(e))

            tock = datetime.now()
            diff = tock - tick
            duration = ms(diff.total_seconds())

            scope_messages = self.__collector.get_all()
            logging.info(f"Received {len(scope_messages)} messages from scope")
            if len(scope_messages) > 0: logging.debug(f"Last 10 messages:\n {''.join(scope_messages[-9:])}")
            self.__collector.reset()

            if result.passed:
                result = self.__validate_test_execution(name=test.name, scoped=scoped, test_data=data,
                                                        scope_messages=scope_messages)

            self.__watcher.test_execution_completed(name=test.name, result=result, scoped=scoped, duration=duration,
                                                    scope_messages=scope_messages, test_data=data)

            logging.info("Finished test %s.", test.name)
