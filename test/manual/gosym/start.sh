#!/bin/bash
set -e

version_regex="^[0-9]+\.[0-9]+$"
start_bash=false
# Check if the version number parameter is provided
if [ $# -eq 0 ]; then
  echo "Please provide a Go version number as an argument"
  echo "For example: ./start.sh 1.20 [bash]"
  exit 1
fi

# Check if the version number matches the expected format
if [[ $1 =~ $version_regex ]]; then
  echo "Valid Go version number: $1"
else
  echo "Invalid Go version number: $1 (should match format X.Y)"
  exit 1
fi

if [ "$2" = "bash" ]; then
  start_bash=true
fi

docker build --tag goversion-"$1" --file Dockerfile --build-arg GO_IMAGE_VER=golang:"$1" .
if [ "$start_bash" = true ] ; then
  docker run --rm -it -v "$(pwd)":/opt/go/output goversion-"$1" bash
else
  docker run -v "$(pwd)":/opt/go/output goversion-"$1"
fi
