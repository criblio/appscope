#!/usr/bin/python3
import ssl
from urllib.request import urlopen
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
with urlopen("https://localhost/", context=ctx) as response:
    content = response.read()
    print(content.decode())
