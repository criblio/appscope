class TestResult:
    def __init__(self, name, wrapped, std_out, std_err, duration):
        self.name = name
        self.wrapped = wrapped
        self.std_err = std_err
        self.std_out = std_out
        self.duration = duration



class Watcher:
    passed_count = 0
    failed_count = 0

    def test_passed(self, result):
        self.passed_count += 1

    def test_failed(self, result):
        self.failed_count += 1


