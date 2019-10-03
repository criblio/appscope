import os
import json
import sys
import subprocess
import time
import watcher as w

help_text = '''options:
--help  display help text
--home  tests home dir (by default /opt/test/ltp/testcases/kernel/syscalls/)
--include_tests include tests array or dir
--exclude_tests exclude tests array or dir
--config config file for tests
'''


class Runner:
    def __init__(self, home, include_tests, exclude_tests, lib_path):
        self.__home = home
        self.__include_tests = include_tests
        self.__exclude_tests = exclude_tests
        self.__test_watcher = w.Watcher()
        self.__lib_path = lib_path

    def run(self):
        self.__test_watcher.start()

        for test in self.__include_tests:
            if test not in self.__exclude_tests:
                if os.path.isdir(test):
                    for d in os.listdir(test):
                        if d not in self.__exclude_tests:
                            self.__find_tests(self.__home + d)
                else:
                    self.__find_tests(self.__home + test)

        self.__test_watcher.finish()
        return (0, 1)[self.__test_watcher.has_failures()]

    def __find_tests(self, test_path):
        if not os.path.isdir(test_path):
            return

        for f in os.listdir(test_path):
            filename = test_path + '/' + f
            if os.path.isfile(filename) and os.access(filename, os.X_OK):
                self.__execute_test(test_path, f, False)
                self.__execute_test(test_path, f, True)

    def __execute_test(self, test_path, test, wrapped):
        start = time.time()
        env = dict(os.environ)

        if wrapped:
            env['LD_PRELOAD'] = self.__lib_path

        p = subprocess.Popen('./' + test, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=test_path, env=env)
        stdout, stderr = p.communicate()

        self.__send_result(w.TestResult(test, wrapped, stdout, stderr, time.time() - start, p.returncode))

    def __send_result(self, result):
        return (self.__test_watcher.test_failed(result), self.__test_watcher.test_passed(result))[
            result.return_code == 0]


def main(args):
    home = "/opt/test/ltp/testcases/kernel/syscalls/"
    include_tests = []
    exclude_tests = []
    lib_path = "/opt/scope/lib/linux/libwrap.so"

    # parse args
    for i in range(len(args)):
        if args[i] == '--help':
            print(help_text)

        if args[i] == '--home':
            home = args[i + 1]

        if args[i] == 'include_tests':
            include_tests = args[i + 1]

        if args[i] == 'exclude_tests':
            exclude_tests = args[i + 1]

        if args[i] == 'lib_path':
            lib_path = args[i + 1]

        if args[i] == '--config':
            path = args[i + 1]
            with open(path, "rb") as f:
                config = json.loads(f.read().decode("utf8"))
                home = config["home"]
                include_tests = config["include_tests"]
                exclude_tests = config["exclude_tests"]
                lib_path = config["lib_path"]

    if not home.endswith('/'):
        home += '/'

    if not os.path.exists(home):
        sys.stderr.write("tests home is not valid")
        return 1

    subprocess.call("make", stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/opt/scope")

    return Runner(home, include_tests, exclude_tests, lib_path).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
