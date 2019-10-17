from abc import abstractmethod, ABC
from typing import Tuple, Any, List

from common import TestResult, TestExecutionData, TestSetResult


class TestExecutionValidator(ABC):

    @property
    def name(self):
        return type(self).__name__

    def should_validate(self, name: str, scoped: bool) -> bool:
        return True

    @abstractmethod
    def validate(self, test_data: Any, scope_messages: List[str]) -> TestResult:
        pass


class TestSetValidator(ABC):

    @property
    def name(self):
        return type(self).__name__

    def should_validate(self, name: str) -> bool:
        return True

    @abstractmethod
    def validate(self, result: TestSetResult) -> TestResult:
        pass


def passed(): return TestResult(passed=True)


def failed(err): return TestResult(passed=False, error=err)


class NoFailuresValidator(TestSetValidator):

    def validate(self, result: TestSetResult) -> TestResult:
        if not result.unscoped_execution_data.result.passed:
            return failed(result.unscoped_execution_data.result.error)

        if not result.scoped_execution_data.result.passed:
            return failed(result.scoped_execution_data.result.error)

        return passed()


class GotTheDataFromScopeValidator(TestSetValidator):

    def validate(self, result: TestSetResult) -> TestResult:
        unscoped_msg_count = len(result.unscoped_execution_data.scope_messages)
        scoped_msg_count = len(result.scoped_execution_data.scope_messages)

        if unscoped_msg_count != 0:
            return failed(f"Expected to have 0 messages but got {unscoped_msg_count}")

        if scoped_msg_count == 0:
            return failed(f"No messages from scope detected")

        return passed()


def validate_all(*assertions: Tuple[bool, str]) -> TestResult:
    for assertions in assertions:
        if not assertions[0]:
            return TestResult(passed=False, error=assertions[1])

    return TestResult(passed=True)


default_test_set_validators = [NoFailuresValidator(), GotTheDataFromScopeValidator()]
