#include <iostream>
#include <cstdlib>


int main() {
   const char* trainImgFile = "../data/train-images";
   const char* trainLabelFile = "../data/train-labels";
   const char* testImgFile= "../data/test-images";
   const char* testLabelFile = "../data/test-labels";

   std::srand(std::time(NULL));

   const char* files[4] = {
      trainImgFile,
      trainLabelFile,
      testImgFile,
      testLabelFile,
   };
   
   NeuralNet net(files);

   net.train(60000 * 6);

   net.evalNetwork(); 

   net.guess();

   return 0;
}
