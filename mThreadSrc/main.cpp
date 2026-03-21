#include <cstdlib>
#include <chrono>
#include "threadedNet.h"

int main() {
   const char* trainImgFile = "../data/train-images";
   const char* trainLabelFile = "../data/train-labels";
   const char* testImgFile= "../data/test-images";
   const char* testLabelFile = "../data/test-labels";

   const size_t batchSize = 10;

   std::srand(std::time(NULL));

   const char* files[4] = {
      trainImgFile,
      trainLabelFile,
      testImgFile,
      testLabelFile,
   };

   NN::NeuralNet net(files);
   net.train();

   return 0;
}
