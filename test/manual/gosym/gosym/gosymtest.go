package main

import "net/http"

func HelloServer(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.Write([]byte("This is an example server.\n"))
}

func testplainServer() {
	http.HandleFunc("/hello", HelloServer)
	http.ListenAndServe("8181", nil)
}

func testplainClient() {
	http.Get("http://www.gnu.org")
}

func testTlsClient() {
	http.Get("https://cribl.io")
}

func testTLSServer() {
	http.HandleFunc("/hello", HelloServer)
	http.ListenAndServeTLS("8182", "server.crt", "server.key", nil)
}

func main() {
	testplainServer()
	testplainClient()
	testTlsClient()
	testTLSServer()
}
