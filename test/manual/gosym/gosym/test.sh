#!/bin/bash
set -e

go_testbinary="/opt/go/gosymtest"
go_version=$(go version | { read -r _ _ v _; echo "${v#go}"; })

# Usage: disassemble_function <function_name> <output_file_name>
disassemble_function() {
    local symbol_name
    local output_file_name
    local output_file_path

    symbol_name=$1
    output_file_name=$2
    
    output_file_path="/opt/go/output/go_${go_version}_${output_file_name}"
    objdump --disassemble="${symbol_name}" $go_testbinary > "${output_file_path}"
}

# 1. Verify pclntab and verify what symbols are available
/opt/go/gosym $go_testbinary -v

# 2. Disassembly the https functions - usefull for comparision
# http2_client_read
disassemble_function "net/http.(*http2clientConnReadLoop).run" http2clientConnReadLoop.run
# http2_client_write
disassemble_function "net/http.http2stickyErrWriter.Write"  http2stickyErrWriter.Write
# http2_server_read
disassemble_function "net/http.(*http2serverConn).readFrames" http2serverConn.readFrames
# http2_server_write
disassemble_function "net/http.(*http2serverConn).Flush" http2serverConn.Flush
# http2_server_preface
disassemble_function "net/http.(*http2serverConn).readPreface" http2serverConn.readPreface
