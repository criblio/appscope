import logging
import sys
from datetime import datetime

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
    parser.add_argument("-id", "--execution_id", help="execution id")
    args = parser.parse_args()

    execution_id = args.execution_id or generate_execution_id(args.target)

    log_level = (logging.INFO, logging.DEBUG)[args.verbose]
    logging.basicConfig(level=log_level)

    logging.info("Execution ID: " + execution_id)
    test_watcher = TestWatcher(execution_id)
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

def generate_execution_id(target):
    return f"{target}:{datetime.now().isoformat()}"


if __name__ == '__main__':
    sys.exit(main())
