from abc import ABC, abstractmethod

from utils import dotdict


class ValidationException(Exception):
    def __init__(self, message, data):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.data = data


class TestResult:
    def __init__(self, passed, test_data=None, error=None):
        self.error = error
        self.test_data = test_data
        self.passed = passed

class TestSetResult:

    def __init__(self):
        self.passed = True
        self.scoped_execution_data = dotdict({
            "result": None,
            "duration": -1,
        })
        self.unscoped_execution_data = dotdict({
            "result": None,
            "duration": -1,
        })
        self.error = None

class Test(ABC):

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def name(self):
        return None

    @staticmethod
    def check(condition, message, data=None):
        if not condition:
            raise ValidationException(message, data)

    def validate_test_set(self, unscoped_test_data, scoped_test_data):
        # this method meant to be overriden
        pass





