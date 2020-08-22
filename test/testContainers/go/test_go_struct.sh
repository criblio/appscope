#!/bin/bash

FAILURE_COUNT=0

# Capture a dump of the .zdebug_info from the elf file
# This contains debug info which describes structures and the offsets
# of their fields.
GO_APP=`pwd`/net/tlsServer
APP_STRUCT_FILE=`pwd`/readelf.dump
readelf --debug-dump=info $GO_APP > $APP_STRUCT_FILE

# What a structure entry looks like in this readelf output:
#      <1><66239>: Abbrev Number: 37 (DW_TAG_structure_type)
#         <6623a>   DW_AT_name        : net/http.connReader
#         <6624e>   DW_AT_byte_size   : 48
#         <6624f>   Unknown AT value: 2900: 25
#         <66250>   Unknown AT value: 2904: 0x74020
#
# What a field entry looks like in this readelf output:
#      <2><66258>: Abbrev Number: 22 (DW_TAG_member)
#         <66259>   DW_AT_name        : conn
#         <6625e>   DW_AT_data_member_location: 0
#         <6625f>   DW_AT_type        : <0x66104>
#         <66263>   Unknown AT value: 2903: 0


function test_structure() {

  STRUCT_NAME=$1
  FIELD_NAME=$2
  FIELD_OFFSET=$3

  # We want to find the line number of the start of our structure
  # and the line number of the start of the next structure
  # so we can limit our search of fields to our current structure

  # What the contents of $STRUCT_LINE_NUMS looks like:
  #      69304: <1><3f584>: Abbrev Number: 37 (DW_TAG_structure_type)
  #      69305-    <3f585>   DW_AT_name        : runtime.g
  #      --
  #      69543: <1><3f8c5>: Abbrev Number: 37 (DW_TAG_structure_type)
  #      69544-    <3f8c6>   DW_AT_name        : runtime.stack
  STRUCT_LINE_NUMS=$(grep -A1 -n DW_TAG_structure_type $APP_STRUCT_FILE | grep -B1 -A3 ": ${STRUCT_NAME}$")

  if [ -z "$STRUCT_LINE_NUMS" ]; then
      echo "  Failed to find structure '${STRUCT_NAME}'"
      FAILURE_COUNT=$(($FAILURE_COUNT + 1))
      return 1
  fi

  # The first field on the first and fourth lines have starting
  # and ending line numbers for our structure
  FIRST_LINE_NUM=$(echo "$STRUCT_LINE_NUMS" | sed -n '1p' | cut -d: -f1)
  LAST_LINE_NUM=$(echo "$STRUCT_LINE_NUMS" | sed -n '4p' | cut -d: -f1)

  # This captures the text of the whole structure
  STRUCT_RAW_OUT=$(cat $APP_STRUCT_FILE | sed -n "${FIRST_LINE_NUM},${LAST_LINE_NUM}p")

  # This captures the text of the whole field out of the structure
  FIELD_OUT=$(echo "$STRUCT_RAW_OUT" | grep -B1 -A3 ": ${FIELD_NAME}$")

  # This tests that the field offset value is as expected
  echo "$FIELD_OUT" | grep -q "DW_AT_data_member_location: ${FIELD_OFFSET}$"
  if [ $? -ne 0 ]; then
      # check for an alternate output format we've also seen:
      # <8e69e>   DW_AT_data_member_location: 2 byte block: 23 30 	(DW_OP_plus_uconst: 48)
      echo "$FIELD_OUT" | grep -q "DW_AT_data_member_location:.*(DW_OP_plus_uconst: ${FIELD_OFFSET})$"
  fi
  if [ $? -ne 0 ]; then
      echo "  Failed to find $FIELD_NAME in $STRUCT_NAME at expected offset $FIELD_OFFSET."
      echo "  What the field looks like:"
      echo "$FIELD_OUT"
      echo ""
      FAILURE_COUNT=$(($FAILURE_COUNT + 1))
  else
      echo "  Found $FIELD_NAME in $STRUCT_NAME at expected offset $FIELD_OFFSET."
  fi
}


function check_expected_file_arg() {
  if [ -z $1 ]; then
      echo "test_go_struct.sh is missing a required arg; the path to an expected structure file"
      echo "Example contents of this file:"
      echo " runtime.g|m=48"
      echo " runtime.m|tls=96"
      echo " net/http.connReader|conn=0"
      echo " net/http.conn|tlsState=48"
      exit 1
  fi

  SCOPE_STRUCT_FILE=$1

  if [ ! -f "$SCOPE_STRUCT_FILE" ]; then
      echo "test_go_struct.sh cannot access $SCOPE_STRUCT_FILE"
      echo "Example contents of $SCOPE_STRUCT_FILE:"
      echo " runtime.g|m=48"
      echo " runtime.m|tls=96"
      echo " net/http.connReader|conn=0"
      echo " net/http.conn|tlsState=48"
      exit 1
  fi
}


function test_each_line() {
  check_expected_file_arg $1

  echo "Using $SCOPE_STRUCT_FILE as the list of what structure offsets scope uses"
  echo "Using $APP_STRUCT_FILE from $GO_APP as the source of truth"

  while read LINE; do
      echo "Evaluating $LINE"
      #format example:"runtime.g|m=48"
      STRUCT_NAME=$(echo $LINE | cut -d'|' -f1)
      FIELD_NAME=$(echo $LINE | cut -d'|' -f2 | cut -d= -f1)
      FIELD_OFFSET=$(echo $LINE | cut -d= -f2)

      test_structure $STRUCT_NAME $FIELD_NAME $FIELD_OFFSET
  done <$SCOPE_STRUCT_FILE
}


echo "executing test_go_struct.sh"
test_each_line $1

if [ $FAILURE_COUNT -eq 0 ]; then
    echo "test_go_struct.sh PASSED"
else
    echo "test_go_struct.sh FAILED"
fi

exit $FAILURE_COUNT
