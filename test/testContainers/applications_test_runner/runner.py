from datetime import datetime

import logging

from common import TestResult, ValidationException
from utils import ms


class Runner:
    def __init__(self, app_controller, tests, watcher, collector):
        self.__collector = collector
        self.__app_controller = app_controller
        self.__tests = tests
        self.__watcher = watcher

    def run(self):

        if not self.__tests:
            logging.warning("No tests provided. Exiting.")
            return 1

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
            results = self.__watcher.get_test_results(test.name)

            if not results.scoped_execution_data.result.passed and results.unscoped_execution_data.result.passed:
                self.__watcher.test_validated(test.name, passed=False, error=results.scoped_execution_data.result.error)
                continue

            unscoped_msg_count = len(results.unscoped_execution_data.scope_messages)
            scoped_msg_count = len(results.scoped_execution_data.scope_messages)

            if unscoped_msg_count != 0:
                self.__watcher.test_validated(test.name, passed=False, error=f"Expected to have 0 messages but got {unscoped_msg_count}")
                continue

            if scoped_msg_count == 0:
                self.__watcher.test_validated(test.name, passed=False, error=f"No messages from scope detected")
                continue

            try:
                test.validate_test_set(results.unscoped_execution_data, results.scoped_execution_data)
                self.__watcher.test_validated(test.name, passed=True)
            except ValidationException as e:
                self.__watcher.test_validated(test.name, passed=False, error=str(e))
                continue

    def __execute_tests(self, scoped):
        for test in self.__tests:
            self.__app_controller.start(scoped)

            logging.info("Running test %s", test.name)
            tick = datetime.now()
            result = None
            try:
                result = test.run()
            except ValidationException as e:
                logging.warning(f"Test {test.name} failed. Message: {str(e)}")
                result = TestResult(passed=False, error=str(e), test_data=e.data)
            except Exception as e:
                logging.warning(f"Test {test.name} failed with exception", exc_info=True)
                result = TestResult(passed=False, error="Exception: " + str(e))

            tock = datetime.now()
            diff = tock - tick
            duration = ms(diff.total_seconds())

            if not self.__app_controller.is_running():
                result = TestResult(passed=False, error="Application is crashed")
                self.__watcher.test_execution_completed(name=test.name, result=result, scoped=scoped, duration=duration,
                                                        scope_messages=[])
                raise Exception("Application is crashed")

            self.__app_controller.stop()

            scope_messages = self.__collector.get_all()
            logging.info(f"Received {len(scope_messages)} messages from scope")
            self.__collector.reset()

            self.__watcher.test_execution_completed(name=test.name, result=result, scoped=scoped, duration=duration,
                                                    scope_messages=scope_messages)

            logging.info("Finished test %s.", test.name)

