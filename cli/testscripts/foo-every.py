#!/usr/bin/python

import time
import sys

i=0
while True:
    out = 'foo %d' % i
    if i % 10 == 0:
        out += ' 10'
    if i % 100 == 0:
        out += ' 100'
    if i % 1000 == 0:
        out += ' 1000'
    if i % 10000 == 0:
        out += ' 10000'
    out += '\n'
    sys.stdout.write(out)
    sys.stdout.flush()
    i=i+1
    # time.sleep(1)
    if i > 1000000:
        sys.exit(0)
