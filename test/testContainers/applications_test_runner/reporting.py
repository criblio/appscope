from typing import Dict
from common import TestSetResult
from tabulate import tabulate


def get_status(result):
    if not result:
        return "-"
    return "fail" if not result.passed else "pass"


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

    print(tabulate(table, headers=["Test", "Result", "Result unscoped/scoped", "Duration unscoped/scoped","Messages uscoped/scoped","Error"]))

