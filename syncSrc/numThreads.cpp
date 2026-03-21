#include <thread>
#include <iostream>
int main() {

   std::cout << std::thread::hardware_concurrency() << '\n';
   return 0;
}
