**Cribl Scope**

# Introduction
Scope represents a project that is part of Deep Collect. As such, Scope is a cpabaility that extracts details from within a running process.
Information is extracted by means of loading a shared library into the address space of a process as the process is started. The library is loaded
by the dynamic linker as part of the normal process of loading libraries in response to an exec system call in Linux/Unix, CreateProcess in Windows.
The mechanism to configure the loader to load our library varies with each OS. In Linux we start with LD_PRELOAD. In macOS we start with
DYLD_INSERT_LIBRARIES and DYLD_FORCE_FLAT_NAMESPACE. In Windows we start with AppInitDLL.

# Requirements
In order to develop code for Scope you will need either a Linux, macOS or Windows environment.

# Install
## macOS
Install gcc and git. These are installed as part of an Apple developer toolkit install. The command below will install the tools without installing the Xcode IDE.
`xcode-select --install`

The tool brew is used to install a number of open source tools that are required.
`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

In order to build any open source code we need additional build tools.
`brew install autoconf automake libtool`

## Linux; RedHat, Oracle, CentOS distros
Install build tools
`yum install git`
`yum install autoconf automake libtool`

## Windows
TBD

# Build
```
git clone git@bitbucket.org:cribl/scope.git
cd scope
make clean
make all
```

# Run
In order to run any given process in a mode where detailed information about the process can be extraced, without changes to code or the executable,
you simply need to run the process with one or more environment variables set. You can set the envirnment variables defined below as needed.
Or, just start the executable with the Scope wrapper. Here's how:
From the top level Scope directory execute the command:
     `% bin/scopeRun your_command params`
Where "your_command" is any executable and "params" are any parameters needed for the executable.  
 
# Developer notes
- The top-level Makefile checks to see what platform you are on and then includes the specific Makefile for the platform from the OS specific directory.
- The run script will determine which platform you are using and will set environment variables accordingly. 

# Environment variables 
## CRIBL_HOME
This envirinment variable defines the base for a Scope install. For a development environment it should be set to the top level dir in the repository.
This variable is set by the run command if a definition does not exist or if it is null. 

## macOS
DYLD_INSERT_LIBRARIES
DYLD_FORCE_FLAT_NAMESPACE
These are the variables that are needed in order for the Scope library to be loaded and enable functions in new processes to be interposed on macOS.

## Linux
LD_PRELOAD
This is the variable that is needed in order for the Scope library to be loaded and enable functions in new processes to be interposed on Linux.

