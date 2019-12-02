import json
import os
from typing import Dict

from tabulate import tabulate

from common import TestSetResult, TestExecutionData
from watcher import TestWatcher


def get_status(result):
    if not result:
        return "-"
    return "fail" if not result.passed else "pass"


def store_results_to_file(watcher: TestWatcher, path: str, scope_version: str):
    def exec_data_to_json(exec_data: TestExecutionData):
        return {
            "status": get_status(exec_data.result),
            "scope_messages": exec_data.scope_messages,
            "error": exec_data.result.error if exec_data.result else None,
            "duration": exec_data.duration,
            "test_data": exec_data.test_data
        }

    with open(os.path.join(path, f"test_results_{watcher.execution_id}.json"), "a") as f:
        results = watcher.get_all_results()
        for test, result in results.items():
            log_row = {
                "id": watcher.execution_id,
                "type": "test_results",
                "test": test,
                "status": get_status(result),
                "scoped_exec_data": exec_data_to_json(result.scoped_execution_data),
                "unscoped_exec_data": exec_data_to_json(result.unscoped_execution_data),
                "error": result.error
            }
            json.dump(log_row, f, default=lambda o: o.__dict__)
            f.write("\n")

        summary_row = {
            "id": watcher.execution_id,
            "type": "summary",
            "scope_version": scope_version,
            "execution_error": watcher.execution_error,
            "status": "pass" if not watcher.has_failures() else "fail",
            "start_date": watcher.start_date.isoformat(),
            "finish_date": watcher.finish_date.isoformat(),
            "total_tests": len(results),
            "failed_tests": len([res for res in results.values() if not res.passed]),
        }

        json.dump(summary_row, f)


def print_summary(results: Dict[str, TestSetResult]):
    def to_table_row(name: str, result: TestSetResult):
        unscoped_msg_count = len(result.unscoped_execution_data.scope_messages)
        scoped_msg_count = len(result.scoped_execution_data.scope_messages)

        return [
            name,
            get_status(result),
            f"{get_status(result.unscoped_execution_data.result)}/{get_status(result.scoped_execution_data.result)}",
            f"{result.unscoped_execution_data.duration}/{result.scoped_execution_data.duration}",
            f"{unscoped_msg_count}/{scoped_msg_count}",
            result.error
        ]

    total_tests = len(results)
    total_failed = len([v for v in list(results.values()) if not v.passed])
    print("Total tests:\t" + str(total_tests))

    print("Total failed:\t" + str(total_failed))
    table = [to_table_row(k, v) for k, v in results.items()]

    print(tabulate(table, headers=["Test", "Result", "Result unscoped/scoped", "Duration unscoped/scoped",
                                   "Messages unscoped/scoped", "Error"]))
