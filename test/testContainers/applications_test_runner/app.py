import logging
import sys

import nginx
from runner import Runner
from watcher import TestWatcher
from reporting import print_summary
import argparse
from scope import ScopeDataCollector, ScopeUDPDataListener
from validation import default_test_set_validators



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-t", "--target", help="target application to test", choices=["nginx"], required=True)
    args = parser.parse_args()
    log_level = (logging.INFO, logging.DEBUG)[args.verbose]
    logging.basicConfig(level=log_level)

    test_watcher = TestWatcher()
    test_watcher.start()
    scope_data_collector = ScopeDataCollector()

    with ScopeUDPDataListener(scope_data_collector, "localhost", 8125):
        try:
            runner = Runner(test_watcher, scope_data_collector)
            runner.add_test_set_validators(default_test_set_validators)
            if args.target == 'nginx':
                nginx.configure(runner)
            runner.run()
            test_watcher.finish()
        except Exception as e:
            logging.error("Tests execution finished with exception: " + str(e), exc_info=True)
            test_watcher.finish_with_error(str(e))

    print_summary(test_watcher.get_all_results())
    logging.shutdown()

    return (0, 1)[test_watcher.has_failures()]


if __name__ == '__main__':
    sys.exit(main())
