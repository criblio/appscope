import argparse
import logging
import sys
from datetime import datetime

import nginx
import splunk
from reporting import print_summary, store_results_to_file
from runner import Runner
from scope import ScopeDataCollector, ScopeUDPDataListener
from validation import default_test_set_validators
from watcher import TestWatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-t", "--target", help="target application to test", choices=["nginx", "splunk"], required=True)
    parser.add_argument("-id", "--execution_id", help="execution id")
    parser.add_argument("-l", "--logs_path", help="path to store the execution results and logs", default="/tmp/")
    parser.add_argument("-s", "--scope_path", help="path to scope application executable",
                        default="/usr/lib/libscope.so")
    args = parser.parse_args()

    execution_id = args.execution_id or generate_execution_id(args.target)
    logs_path = args.logs_path

    log_level = (logging.INFO, logging.DEBUG)[args.verbose]
    logging.basicConfig(level=log_level)

    logging.info("Execution ID: " + execution_id)
    logging.info("Logs path: " + logs_path)
    test_watcher = TestWatcher(execution_id)
    test_watcher.start()
    scope_data_collector = ScopeDataCollector()

    with ScopeUDPDataListener(scope_data_collector, "localhost", 8125):
        try:
            runner = Runner(test_watcher, scope_data_collector)
            runner.add_test_set_validators(default_test_set_validators)
            if args.target == 'nginx':
                nginx.configure(runner, args)
            if args.target == 'splunk':
                splunk.configure(runner, args)
            runner.run()
            test_watcher.finish()
        except Exception as e:
            logging.error("Tests execution finished with exception: " + str(e), exc_info=True)
            test_watcher.finish_with_error(str(e))

    print_summary(test_watcher.get_all_results())
    store_results_to_file(watcher=test_watcher, path=logs_path)
    logging.shutdown()

    return (0, 1)[test_watcher.has_failures()]


def generate_execution_id(target):
    return f"{target}:{datetime.now().isoformat()}"


if __name__ == '__main__':
    sys.exit(main())
