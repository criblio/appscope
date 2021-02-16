#!/usr/bin/python3

f = open('wontsee.txt', 'w')
f.write('wont see this string')
f.close()

f = open('willsee.log', 'w')
f.write('will see this string')
f.close()
