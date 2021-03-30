#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

scope run -p -- curl --http1.1 -so /dev/null https://cribl.io/
cwd=$(pwd); cd $(scope hist -d) && go-bindata -pkg flows -o ${DIR}/flow_file_test.go -fs . payloads; cd -
