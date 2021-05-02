# ---
# Cribl AppScope™ - Build under Docker
#
# by Paul Dugas <pdugas@cribl.io>
#

# ---
# Use the current Ubuntu image by default. Override this like below.
#
#     docker build --build-arg IMAGE=ubuntu:groovy ...
#
ARG IMAGE=ubuntu
FROM $IMAGE

# ---
# Use the current version of Go in the longsleep/golang-backports PPA by
# default. Override this like below.
#
#     docker build --build-arg GOLANG=golang-1.16 ...
#
ARG GOLANG=golang
ENV GOLANG=$GOLANG

# These are needed to bypass the prompts when tzdata is installed.
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"

# ---
# Install packages. Note extra PPA for Go.
#
RUN apt-get update && \
    apt-get install -y software-properties-common gpg apt-utils && \
    add-apt-repository ppa:longsleep/golang-backports && \
    apt-get update && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 52B59B1571A79DBC054901C0F6BC817356A3D45E && \
    apt-get install -y $GOLANG git autoconf automake libtool cmake lcov upx make && \
    apt-get clean

# ---
# The local git clone of the project is mounted as /root/appscope. See Makefile.
#
#     docker run -v $(pwd):/root/appscope ...
#
WORKDIR /root/appscope

# fini
