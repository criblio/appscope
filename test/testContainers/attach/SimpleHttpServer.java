import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

public class SimpleHttpServer {
  static class StatusHandler implements HttpHandler {
    public void handle(HttpExchange exchange) throws IOException {
      exchange.getResponseHeaders().set("Custom-Header-1", "A~HA");
      exchange.sendResponseHeaders(200, "OK\n".length());
      OutputStream os = exchange.getResponseBody();
      os.write("OK\n".getBytes());
      os.flush();
      os.close();
      exchange.close();
    }
  }

  public static void main(String[] args) throws IOException {
    HttpServer server = HttpServer.create(new InetSocketAddress(8000), 1024);

    server.createContext("/status", new StatusHandler());

    server.start();
  }
}
