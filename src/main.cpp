#include <iostream>

#include "mnistparser.h"
class NeuralNet {
   private:
      std::vector<uint8_t> trainImgs;
      std::vector<uint8_t> trainLabels;
      std::vector<uint8_t> testImgs;
      std::vector<uint8_t> testLabels;

   public:
      NeuralNet(const char** files) {
         MnistParser trainParser(files[0], files[1], trainImgs, trainLabels);
         MnistParser testParser(files[2], files[3], testImgs, testLabels);
      }
};

int main() {
   const char* trainImgFile = "../data/train-images";
   const char* trainLabelFile = "../data/train-labels";
   const char* testImgFile= "../data/test-images";
   const char* testLabelFile = "../data/test-labels";

   const char* files[4] = {
      trainImgFile,
      trainLabelFile,
      testImgFile,
      testLabelFile,
   };
   
   NeuralNet net(files);

   return 0;
}
