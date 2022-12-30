# IPC Notes

- For IPC communication we are using pair of posix message queue:

- The client in the communication is `scope`
- The server in the communication is scoped application
- The message queue created by `scope` is `ScopeIPCOut.<PID>` and `ScopeIPCIn.<PID>`
- scoped application reads from `ScopeIPCIn.<PID>` and writes to `ScopeIPCOut.<PID>`
- `scope` reads from `ScopeIPCOut.<PID>` and writes to `ScopeIPCIn.<PID>`

![IPC Demo](images/ipc.gif)

# IPC adding new request

Depending on expected logic adding new request required adding handling both on CLI side and library side.

- Adding new request, CLI: `ipcscope.go`
- Adding new request, library: `ipc_scope_req_t` in `ipc.c`
- Adding response for the request, CLI: `ipcscope.go`
- Adding response for the request, library: `*supportedResp` in `ipc.c`
- (Optional) Adding processing request in the library see `ipcProcessSetCfg` as an example
