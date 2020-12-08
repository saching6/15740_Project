#include <torch/script.h>

#include <iostream>
#include <memory>

int main( int argc, const char* argv[] ){
   if (argc != 2) {
      std::cerr << "usage: example-app <path-to-exported-script-module>\n";
      return -1;
   }

   std::cout << "Hello, World\n";
   return 0;
}
