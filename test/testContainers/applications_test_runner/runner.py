from datetime import datetime
import sys
import logging
import nginx
from watcher import TestWatcher
from common import TestResult, ValidationException, TestSetResult
from utils import ms
from reporting import print_summary
import argparse


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
        self.__watcher.start()
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

        self.__watcher.finish()

        print_summary(self.__watcher.get_all_results())

        return (0, 1)[self.__watcher.has_failures()]


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
            self.__watcher.test_execution_completed(name=test.name, result=result, scoped=scoped, duration=duration)
            logging.info("Finished test %s.", test.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    log_level = (logging.INFO, logging.DEBUG)[args.verbose]
    logging.basicConfig(level=log_level)

    app_controller = nginx.NginxApplicationController()
    test_watcher = TestWatcher()
    return_code = Runner(app_controller, nginx.get_tests(), test_watcher).run()
    logging.shutdown()
    return return_code


if __name__ == '__main__':
    sys.exit(main())
