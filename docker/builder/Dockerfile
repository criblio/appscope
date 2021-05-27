# ---
# Cribl AppScope™ - Build under Docker
#
# by Paul Dugas <paul@dugas.cc>
#

# ---
# Use the Ubuntu 18.04 image by default. Override this like below.
#
#     docker build --build-arg IMAGE=ubuntu:latest ...
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

# ---
# Install packages. Note extra PPA for Go.
#
# Also note that this logic duplicates the top-level `install_build_tools.sh`
# script. Trying to use that script here causes issues because of its use of
# `sudu` when running the package installer. It clears the environment so the
# additions above don't apply and we end up with interaction when some of the
# dependencies (i.e. tzdata) are installed.
#
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"
RUN apt-get update && \
    apt-get install -y software-properties-common gpg apt-utils && \
    add-apt-repository ppa:longsleep/golang-backports && \
    apt-get update && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 52B59B1571A79DBC054901C0F6BC817356A3D45E && \
    apt-get install -y $GOLANG git autoconf automake libtool cmake lcov upx make gdb vim emacs strace lsof sudo && \
    apt-get clean && \
    sed -i 's/%sudo	ALL=(ALL:ALL) ALL/%sudo	ALL=(ALL:ALL) NOPASSWD:ALL/' /etc/sudoers

# ---
# Create a user 
#
# The Makefile passes the current user's UID/GID in as these ARGs when
# building. We then run using these IDs so the ownership of the mounted files
# don't get hosed.
#
ARG UID=1000
ARG GID=1000
RUN addgroup --gid $GID appscope && \
    adduser --home /opt/appscope --uid $UID --gid $GID appscope && \
    usermod -a -G sudo appscope

# ---
# The local git clone of the project is mounted as /root/appscope. See Makefile.
#
#     docker run -v $(pwd):/root/appscope ...
#
WORKDIR /opt/appscope

# fini
