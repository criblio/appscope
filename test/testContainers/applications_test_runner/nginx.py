from typing import Any, List

from common import TestResult
from proc import SubprocessAppController
from runner import Runner
from validation import TestExecutionValidator, validate_all
from web import TestGetUrl, TestPostToUrl


class NetworkMetricsCollectedValidator(TestExecutionValidator):

    def __init__(self, applicable_tests: List[str]):
        self.applicable_tests = applicable_tests

    def should_validate(self, name: str, scoped: bool) -> bool:
        return scoped and name in self.applicable_tests

    def validate(self, test_data: Any, scope_messages: List[str]) -> TestResult:
        return validate_all(
            (any("net.tx" in msg for msg in scope_messages if "#proc:nginx" in msg), "No 'net.tx' metrics is collected"),
            (any("net.rx" in msg for msg in scope_messages if "#proc:nginx" in msg), "No 'net.rx' metrics is collected")
        )


def configure(runner: Runner):
    runner.set_app_controller(SubprocessAppController(["nginx", "-g", "daemon off;"], "nginx"))
    get_home_page = TestGetUrl(url="http://localhost/", requests=10000)
    post_file = TestPostToUrl(url="http://localhost/log/", requests=10000, post_file="./post.json")
    runner.add_test_execution_validators([NetworkMetricsCollectedValidator([get_home_page.name, post_file.name])])
    runner.add_tests([get_home_page, post_file])
