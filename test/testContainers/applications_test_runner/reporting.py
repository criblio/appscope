from typing import Dict
from common import TestSetResult
from tabulate import tabulate
import json


def to_status(passed):
    return ("fail", "pass")[passed]


def print_summary(results: Dict[str, TestSetResult]):
    def to_table_row(name: str, result: TestSetResult):
        return [
            name,
            to_status(result.passed),
            to_status(result.unscoped_execution_data.result.passed),
            to_status(result.scoped_execution_data.result.passed),
            result.unscoped_execution_data.duration,
            result.scoped_execution_data.duration,
        ]

    total_tests = len(results)
    total_failed = len([v for v in list(results.values()) if not v.passed])
    print("Total tests:\t" + str(total_tests))

    print("Total failed:\t" + str(total_failed))
    table = [to_table_row(k, v) for k, v in results.items()]

    print(tabulate(table, headers=["Test", "Result", "Result unscoped", "Result scoped", "Duration unscoped",
                                   "Duration scoped"]))
