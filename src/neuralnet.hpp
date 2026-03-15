#include "mnistparser.hpp"
#include <cmath>
#include <armadillo>

class NeuralNet {
   private:

      static constexpr float ETA = 0.01;
      static const size_t LAYERS = 4;
      static const size_t HIDDEN_LAYER_DIM = 16;
      static const size_t numInputs = 28 * 28;
      static const size_t numOutputs = 10;

      std::vector< arma::Mat<float> > weights;
      std::vector< arma::Col<float> > biases;
      std::vector< arma::Col<float> > neuronsZ;
      std::vector< arma::Col<float> > neuronsA;

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
            // Xavier Initialization
            weights[i] = arma::Mat<float>(neuronVecDims[i+1], neuronVecDims[i], arma::fill::randn) * std::sqrt(1.0f / neuronVecDims[i]);

            biases[i] = arma::Col<float>(neuronVecDims[i+1], arma::fill::zeros);
         }
 
      }

   public:

      std::vector<uint8_t> trainImgs;
      std::vector<uint8_t> trainLabels;
      std::vector<uint8_t> testImgs;
      std::vector<uint8_t> testLabels;

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

      arma::Col<float> layerGrad(const arma::Col<float>& errorL, size_t layerNum) {
         arma::Col<float> currLayerError = errorL % (neuronsA[layerNum] % ( 1 - arma::Col<float>(neuronsA[layerNum])) );
         arma::Mat<float> weightGrad = currLayerError * neuronsA[layerNum - 1].t();

         weights[layerNum - 1] = weights[layerNum - 1] - (weightGrad * ETA);
         biases[layerNum - 1] = biases[layerNum - 1] - (currLayerError * ETA); 

         return weights[layerNum - 1].t() * currLayerError;
      }

      arma::Col<float>& computeCostGrad(arma::Col<float> goal) {
         arma::Col<float> initGrad = 2 * (neuronsA[LAYERS - 1] - goal);
         arma::Col<float> tempGrad;
          
         for (size_t i = LAYERS - 1; i > 0; i--) {
            tempGrad = layerGrad(( i == LAYERS - 1 ? initGrad : tempGrad ), i);
         }

         return neuronsA[LAYERS - 1]; 
      }

      void placeImageInNet(size_t imageNum, std::vector<uint8_t> &imgsRef) {
         std::vector<uint8_t> individualImage(imgsRef.begin() + imageNum * numInputs, imgsRef.begin() + imageNum * numInputs + numInputs); 
         neuronsZ[0] = arma::conv_to< arma::fmat >::from(individualImage);
         neuronsZ[0] = neuronsZ[0] % arma::Col<float>(numInputs, arma::fill::value(1.0f / 255));
         neuronsA[0] = neuronsZ[0];
      }

      void train(size_t steps) {
         static arma::Col<float> result(10);
         arma::Col<float> goal(10, arma::fill::zeros);
         static arma::Col<float> costVec(10, arma::fill::zeros);

         static size_t batchLocation = rand() % 60000;
         size_t currSteps = 0;

         while (currSteps != steps) {
            this->placeImageInNet(batchLocation, this->trainImgs);

            goal[ this->trainLabels[batchLocation] ] = 1;

            this->feedForward();
            result = this->computeCostGrad(goal);
            
            costVec = (result - goal) % (result - goal);
            double cost = arma::accu(costVec);

            goal.zeros();
            batchLocation = rand() % 60000;
            currSteps++;
         }
      }

      void evalNetwork() {
         double diff = 0;
         arma::Col<float> goal(10, arma::fill::zeros);

         size_t i = 0;
         for ( ; i < 10000; i++) {
            this->placeImageInNet(i, this->testImgs);
            this->feedForward();

            goal[this->testLabels[i]] = 1;
            
            diff += arma::accu((this->neuronsA[LAYERS - 1] - goal) % (this->neuronsA[LAYERS - 1] - goal));
            goal.zeros();
         }

         std::printf("%s", "average cost across test examples: ");
         std::printf("%f", diff / i);
         std::printf("%c", '\n');
      }

      void guess() {
         int input[5];

         for (int i = 0; i < 5; i++) {
            input[i] = rand() % 10000;
         }

         for (int j = 0; j < 5; j++) {
            this->placeImageInNet(input[j], this->testImgs);

            std::printf("%s", "----------------------------"); 
            for (int i = input[j] * 28*28; i < (input[j]+1) * 28*28; i++) {
               if (i % 28 == 0) std::printf("%c", '\n');
               this->testImgs[i] < 0.4 ? std::printf("%s", ". ") : std::printf("%s", "# ");
            }
            this->feedForward();
            std::printf("%s", "\n----------------------------\n");

            std::printf("%s", "Goal: "); std::printf("%i", this->testLabels[input[j]]); std::printf("%c", '\n'); 

            int maxIndex = 0;
            for (int i = 1; i < this->neuronsA[LAYERS - 1].n_elem - 1; i++) {
               if (this->neuronsA[LAYERS-1][i] > this->neuronsA[LAYERS-1][maxIndex]) maxIndex = i;
            }

            std::printf("%s", "Guess: "); std::printf("%i", maxIndex); std::printf("%c", '\n');
         }
      }
};
