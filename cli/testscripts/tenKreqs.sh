#!/bin/bash
for i in `seq 1 10000`; do 
    curl http://localhost:8001/log.py >/dev/null 2>/dev/null
done