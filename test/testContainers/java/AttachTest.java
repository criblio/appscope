package io.cribl.scope;
import java.io.IOException;

public class AttachTest {
  void firstprint(String str) {
    System.out.println("Print First From Attach Test" + str);
  }
  
  void second_print(String str) {
    System.out.println("Print Second From Attach Test" + str);
  }

  static void attach_dummy_print(String str) {
    System.out.println("Hello from dummy print");
    System.out.println(str + " Dummy end");
  }
}