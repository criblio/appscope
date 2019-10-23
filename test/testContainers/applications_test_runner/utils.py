import string
from random import choice


def ms(seconds):
    return int(seconds * 1000)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def extract_num(string):
    return [int(s) for s in string.split() if s.isdigit()]


def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(choice(letters) for i in range(stringLength))
