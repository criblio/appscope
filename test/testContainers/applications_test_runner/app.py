import logging
import sys

import nginx
from runner import Runner
from watcher import TestWatcher
from reporting import print_summary
import argparse
from scope import ScopeDataCollector, ScopeUDPDataListener


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    log_level = (logging.INFO, logging.DEBUG)[args.verbose]
    logging.basicConfig(level=log_level)

    app_controller = nginx.NginxApplicationController()
    test_watcher = TestWatcher()
    test_watcher.start()
    scope_data_collector = ScopeDataCollector()

    with ScopeUDPDataListener(scope_data_collector, "localhost", 8125):
        try:
            Runner(app_controller, nginx.get_tests(), test_watcher, scope_data_collector).run()
            test_watcher.finish()
        except Exception as e:
            logging.error("Tests execution finished with exception: " + str(e))
            test_watcher.finish_with_error(str(e))

    print_summary(test_watcher.get_all_results())
    logging.shutdown()

    return (0, 1)[test_watcher.has_failures()]


if __name__ == '__main__':
    sys.exit(main())
