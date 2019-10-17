from abc import ABC, abstractmethod
from typing import Any, Tuple, List


class TestResult:
    passed: bool
    error: str

    def __init__(self, passed, error=None):
        self.error = error
        self.passed = passed


class TestExecutionData:
    result: TestResult
    duration: int
    scope_messages: List[str]
    test_data: Any

    def __init__(self):
        self.scope_messages = []
        self.duration = 0
        self.test_data = None


class TestSetResult:
    error: None
    unscoped_execution_data: TestExecutionData
    scoped_execution_data: TestExecutionData
    passed: bool

    def __init__(self):
        self.passed = False
        self.scoped_execution_data = TestExecutionData()
        self.unscoped_execution_data = TestExecutionData()
        self.error = None


class AppController(ABC):

    @abstractmethod
    def start(self, scoped):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def is_running(self):
        pass

    @property
    @abstractmethod
    def name(self):
        return None


class Test(ABC):

    @abstractmethod
    def run(self) -> Tuple[TestResult, Any]:
        pass

    @property
    @abstractmethod
    def name(self):
        return None
