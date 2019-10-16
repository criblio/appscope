import logging

import nginx
from runner import Runner
from watcher import TestWatcher
import sys
from reporting import print_summary
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    log_level = (logging.INFO, logging.DEBUG)[args.verbose]
    logging.basicConfig(level=log_level)

    app_controller = nginx.NginxApplicationController()
    test_watcher = TestWatcher()
    test_watcher.start()

    try:
        Runner(app_controller, nginx.get_tests(), test_watcher).run()
        test_watcher.finish()
    except Exception as e:
        logging.error("Tests execution finished with exception: " + str(e))
        test_watcher.finishWithError(str(e))

    print_summary(test_watcher.get_all_results())
    logging.shutdown()

    return (0, 1)[test_watcher.has_failures()]


if __name__ == '__main__':
    sys.exit(main())