#
# GDB Setup for AppScope
#

# --
# This is here so when this directory is mounted as the user's home directory
# under `make docker-*` GDB will be configured they way we like it.
#
# You'll get a warning about auto-loading being declined when you run GDB in
# this directory unless you add something like this to ~/.gdbinit in your
# home directory.
#
#     set auto-load safe-path ~/Projects/*/*
# --

# Preload the library assuming we're in the top of a built working copy
set environment LD_PRELOAD=./src/.libs/libscope.so

# Allow breakpoints on function not yet found without confirmation
set breakpoint pending on

# Set this when crashing in a child process
#set follow-fork-mode child

# Save command history
set history filename ~/.gdb_history
set history save on
set history remove-duplicates unlimited
