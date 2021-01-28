# § Using AppScope

# Using the CLI

You can use the AppScope CLI to monitor applications and other processes, as shown in these examples:

Generate a large data set:

`scope firefox`

Scope every subsequent shell command:

`scope bash` 

If you have [Cribl LogStream](https://cribl.io/download/) installed, try:

`scope /<cribl_home>/bin/cribl server`

To release a process, like `scope bash`, `exit` the process in the shell. If you do this with the bash shell itself, you won't be able to `scope bash` again in the same terminal session.
