package io.cribl.scope;

class JavaTest {
  public static final String staticField = "static";
  private String field1;
  public Integer field2;

  void print(String str) {
    System.out.println(str);
  }

  native void nativeMethod(String str);
  
  public static void main (String args[]) {
    JavaTest test = new JavaTest();
    test.print("Hello World");
  }

}