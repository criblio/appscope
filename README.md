# Introduction
Scope represents a project that is part of Deep Collect. As such, Scope is a cpabaility that extracts details from within a running process. Information is extracted by means of loading a shared library into the address space of a process as the process is started. The library is loaded by the dynamic linker as part of the normal process of loading libraries in response to an exec system call in Linux/Unix, CreateProcess in Windows. The mechanism to confure the loader to load our library varies with each OS. In Linux we start with LD_PRELOAD. In macOS we start with DYLD_INSERT_LIBRARIES and DYLD_FORCE_FLAT_NAMESPACE. In Windows we start with AppInitDLL.

# Requirements
In order to develop code for Scope you will need either a Linux, macOS or Windows environment.

# Install
macOS
Install gcc and git. These installed as part of an Apple developer tool install. The command below will install the tools without also installing the Xcode IDE.
xcode-select --install

The tool brew is used to install a number of open source tools that are required.
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

In order to build any open source code we need additional build tools.
brew install autoconf automake libtool

Linux; RedHat, Oracle, CentOS distros
Install build tools
yum install git
yum install autoconf automake libtool

Windows
TBD

# Build
git clone git@bitbucket.org:cribl/scope.git
cd scope
make all

# Run
TBD
 
# Developer notes
TBD

# Environment variables 
TBD

