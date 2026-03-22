#include <iostream>
#include <cstdlib>
#include <chrono>
#include "neuralnet.hpp"

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

   NeuralNet net(files);

   net.train<10>();

   std::printf("Avg cost: %f", net.evalNetwork()); 

   return 0;
}
