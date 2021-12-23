#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TEST_FILE "file.txt"

int main () {
   FILE *fp;
   int c;
   char str[] = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
  
   fp = fopen(TEST_FILE,"w");
   if (!fp)
      return EXIT_FAILURE;

   fwrite(str , sizeof(char), sizeof(str), fp);

   fclose(fp);
   unlink(TEST_FILE);
   return EXIT_SUCCESS;
}
