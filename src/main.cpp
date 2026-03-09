#include <iostream>

#include "mnistparser.hpp"
#include <cmath>
#include <armadillo>
class NeuralNet {
   private:
      std::vector<uint8_t> trainImgs;
      std::vector<uint8_t> trainLabels;
      std::vector<uint8_t> testImgs;
      std::vector<uint8_t> testLabels;

      static constexpr float ETA = 0.1;
      static const size_t LAYERS = 4;
      static const size_t HIDDEN_LAYER_DIM = 16;
      static const size_t numInputs = 28 * 28;
      static const size_t numOutputs = 10;

      std::vector< arma::Mat<float> > weights;
      std::vector< arma::Col<float> > biases;
      std::vector< arma::Col<float> > neuronsZ;
      std::vector< arma::Col<float> > neuronsA;

      arma::Col<float> costGrad;

      size_t neuronVecDims[LAYERS];

      void initNetwork() {
         neuronVecDims[0] = numInputs;
         neuronVecDims[LAYERS - 1] = numOutputs;

         for (size_t i = 1; i < LAYERS - 1; i++) { 
            neuronVecDims[i] = HIDDEN_LAYER_DIM; 
         }

         for (size_t i = 0; i < LAYERS; i++) { 
            neuronsA[i] = arma::Col<float>(neuronVecDims[i], arma::fill::randu); 
            neuronsZ[i] = arma::Col<float>(neuronVecDims[i], arma::fill::randu); 
         }

         for (size_t i = 0; i < LAYERS - 1; i++) {
            std::cout << "Dims: " << neuronVecDims[i+1] << "x" << neuronVecDims[i] << '\n';
            weights[i] = arma::Mat<float>(neuronVecDims[i+1], neuronVecDims[i], arma::fill::randn);
            biases[i] = arma::Col<float>(neuronVecDims[i+1], arma::fill::randn);
         }
 
      }

   public:
      NeuralNet(const char** files) : 
         weights(LAYERS - 1),
         biases(LAYERS - 1),
         neuronsA(LAYERS),
         neuronsZ(LAYERS)
      {
         MnistParser trainParser(files[0], files[1], trainImgs, trainLabels);
         MnistParser testParser(files[2], files[3], testImgs, testLabels);

         initNetwork();
      }

      void feedForward() {
         for (size_t i = 0; i < LAYERS - 1; i++) {
            neuronsZ[i+1] = (weights[i] * neuronsA[i]) + biases[i];
            neuronsA[i+1] = neuronsZ[i+1]; 
            neuronsA[i+1].transform([] (float val) {
                  return 1.0 / (1.0f + std::exp(-val));
               });
         }
      }

      arma::Col<float> layerGrad(const arma::Col<float>& currLayerGrad, size_t layerNum) {
         arma::Col<float> activationGradient(currLayerGrad.n_elem);

         arma::Col<float> currLayerError = currLayerGrad % (neuronsA[layerNum] % ( 1 - arma::Col<float>(neuronsA[layerNum])) );
         arma::Mat<float> weightGrad = currLayerError * neuronsA[layerNum - 1].t();

         weights[layerNum - 1] = weights[layerNum - 1] - (weightGrad * (ETA * -1.0f));
         biases[layerNum - 1] = biases[layerNum - 1] - (currLayerError * (ETA * -1.0f)); 

         return weightGrad.t() * currLayerError; //next layer's gradient
      }

      arma::Col<float>& computeCostGrad(arma::Col<float> goal) {
         arma::Col<float> initGrad = (neuronsZ[LAYERS - 1] - goal);
         initGrad = initGrad % initGrad; //squaring element-wise
          
         for (size_t i = LAYERS - 1; i > 0; i--) {
            arma::Col<float> tempGrad = layerGrad(( i == LAYERS - 1 ? initGrad : tempGrad ), i);
         }

         return neuronsA[LAYERS - 1]; 
      }

      void placeImageInNet(size_t imageNum) {
         std::vector<uint8_t> individualImage(trainImgs.begin() + imageNum * numInputs, trainImgs.begin() + imageNum * numInputs + numInputs); 
         neuronsZ[0] = arma::conv_to< arma::fmat >::from(individualImage);
         neuronsA[0] = neuronsZ[0];
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

   net.placeImageInNet(0);
   
   arma::Col<float> goal(10, arma::fill::zeros);
   goal[3] = 1;

   arma::Col<float> res(10);

   for (int i = 0; i < 1000; i++) {
      net.feedForward();
      res = net.computeCostGrad(goal);
      std::cout << res[3] << '\n';
   }
   return 0;
}
