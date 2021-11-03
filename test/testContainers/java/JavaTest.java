package io.cribl.scope;
import java.io.IOException;

public class JavaTest {
  void firstprint(String str) {
    System.out.println(str);
  }
  
  void second_print(String str) {
    System.out.println("Print Second " + str);
  }

  public static void main (String args[]) {
    JavaTest test = new JavaTest();
    test.firstprint("Hello World");
    try {
      char ch = (char) System.in.read();
    }
    catch(IOException e) {
      e.printStackTrace();
    }
    test.firstprint("Bye World");
  }
}