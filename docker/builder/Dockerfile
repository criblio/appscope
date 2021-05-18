# ---
# Cribl AppScope™ - Build under Docker
#
# by Paul Dugas <paul@dugas.cc>
#

# ---
# Use the current Ubuntu image by default. Override this like below.
#
#     docker build --build-arg IMAGE=ubuntu:groovy ...
#
ARG IMAGE=ubuntu:18.04
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
# Also note that this logic duplicates the top-level `install_build_tools.sh`
# script. Trying to use that script here causes issues because of its use of
# `sudu` when running the package installer. It clears the environment so the
# additions above don't apply and we end up with interaction when some of the
# dependencies (i.e. tzdata) are installed.
#
RUN apt-get update && \
    apt-get install -y software-properties-common gpg apt-utils && \
    add-apt-repository ppa:longsleep/golang-backports && \
    apt-get update && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 52B59B1571A79DBC054901C0F6BC817356A3D45E && \
    apt-get install -y $GOLANG git autoconf automake libtool cmake lcov upx make gdb vim emacs strace lsof && \
    apt-get clean

# ---
# Extra setup for use with `make docker-run`
#
RUN echo "set environment LD_PRELOAD=/root/appscope/lib/linux/libscope.so" >> /root/.gtbinit && \
    echo "set breakpoint pending on" >> /root/.gtbinit && \
    echo "set follow-fork-mode child" >> /root/.gtbinit

# ---
# The local git clone of the project is mounted as /root/appscope. See Makefile.
#
#     docker run -v $(pwd):/root/appscope ...
#
WORKDIR /root/appscope

# fini
