from datetime import datetime

import logging

from common import TestResult, ValidationException
from utils import ms



class Runner:
    def __init__(self, app_controller, tests, watcher):
        self.__app_controller = app_controller
        self.__tests = tests
        self.__watcher = watcher

    def run(self):

        if not self.__tests:
            logging.warning("No tests provided. Exiting.")
            return 1

        logging.info("Starting unscoped tests.")
        logging.info("Starting app %s in unscoped mode.", self.__app_controller.name)
        self.__app_controller.start(scoped=False)
        self.__execute_tests(False)
        logging.info("Unscoped tests execution finished. Stopping app.")
        self.__app_controller.stop()

        logging.info("Starting scoped tests.")
        logging.info("Starting app %s in scoped mode.", self.__app_controller.name)
        self.__app_controller.start(scoped=True)
        self.__execute_tests(True)
        logging.info("Scoped tests execution finished. Stopping app.")
        self.__app_controller.stop()

        logging.info("Validating results.")
        self.__validate_results()

    def __validate_results(self):
        for test in self.__tests:
            results = self.__watcher.get_test_results(test.name)

            if not results.scoped_execution_data.result.passed and results.unscoped_execution_data.result.passed:
                self.__watcher.test_validated(test.name, passed=False, error=results.scoped_execution_data.result.error)
                continue

            try:
                test.validate_test_set(results.unscoped_execution_data, results.scoped_execution_data)
                self.__watcher.test_validated(test.name, passed=True)
            except ValidationException as e:
                self.__watcher.test_validated(test.name, passed=False, error=str(e))
                continue

    def __execute_tests(self, scoped):
        for test in self.__tests:
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
                self.__watcher.test_execution_completed(name=test.name, result=result, scoped=scoped, duration=duration)
                raise Exception("Application is crashed")

            self.__watcher.test_execution_completed(name=test.name, result=result, scoped=scoped, duration=duration)
            logging.info("Finished test %s.", test.name)



