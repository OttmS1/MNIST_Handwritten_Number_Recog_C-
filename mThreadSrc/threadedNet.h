#include <algorithm> //std::ranges::shuffle
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <random>
#include <thread>
#include <utility> //std::pair
#include <vector>

#include <armadillo>

#include "mnistparser.hpp"
namespace NN {

typedef std::pair<uint8_t, std::vector<uint8_t>> ILPair;

static constexpr size_t numInputs = 28 * 28;
static const size_t numOutputs = 10;
static const size_t LAYERS = 4;

class NeuralNet;

enum TrainerState { STOP, IDLE, RUN };

struct Trainer {
  // Neural Network data
  NeuralNet *net;
  size_t batchSize;
  size_t batchLocation;
  std::vector<std::vector<ILPair>> *batches;
  std::vector<arma::Col<float>> neuronsZ;
  std::vector<arma::Col<float>> neuronsA;
  std::vector<arma::Mat<float>> weightChanges;
  std::vector<arma::Col<float>> biasChanges;
  arma::Col<float> goal;

  // thread data
  std::thread thisThread;
  std::atomic<int> state;

  Trainer() {}
  Trainer(NeuralNet *thisNet, std::vector<std::vector<ILPair>> *_batches,
          const size_t _batchSize);
  ~Trainer();
  Trainer(const Trainer &) = delete;
  Trainer &operator=(const Trainer &) = delete;
  Trainer(Trainer &&) = delete;
  Trainer &operator=(Trainer &&) = delete;

  void listener();
  void startTraining();

  void resetWeightChanges() {
    for (arma::Mat<float> &mat : weightChanges) {
      mat.zeros();
    }
  }
  void resetBiasChanges() {
    for (arma::Col<float> &col : biasChanges) {
      col.zeros();
    }
  }

}; // struct Trainer

class NeuralNet {
private:
  std::vector<ILPair> train_imgLabelPairs;
  std::vector<ILPair> test_imgLabelPairs;

  static constexpr float ETA = 0.01f;
  static const size_t HIDDEN_LAYER_DIM = 128;

  std::vector<arma::Mat<float>> weights;
  std::vector<arma::Col<float>> biases;
  std::vector<arma::Mat<float>> weightChanges;
  std::vector<arma::Col<float>> biasChanges;
  std::vector<arma::Col<float>> neuronsZ;
  std::vector<arma::Col<float>> neuronsA;

  size_t neuronVecDims[LAYERS];

  void initNetwork();

public:
  NeuralNet(const char **files);

  void constructLocalNeurons(std::vector<arma::Col<float>> &neuronsZ,
                             std::vector<arma::Col<float>> &neuronsA);
  void constructLocalChanges(std::vector<arma::Mat<float>> &weightChanges,
                             std::vector<arma::Col<float>> &biasChanges);

  void placeImageInNet(std::vector<uint8_t> &img,
                       std::vector<arma::Col<float>> &neuronsZ,
                       std::vector<arma::Col<float>> &neuronsA);

  void feedForward(std::vector<arma::Col<float>> &neuronsZ,
                   std::vector<arma::Col<float>> &neuronsA);

  arma::Col<float> layerGrad(const arma::Col<float> &errorL,
                             const size_t layerNum,
                             std::vector<arma::Col<float>> &neuronsA,
                             std::vector<arma::Mat<float>> &weightChanges,
                             std::vector<arma::Col<float>> &biasChanges,
                             Trainer *trainer);

  arma::Col<float> &backprop(arma::Col<float> &goal,
                             std::vector<arma::Col<float>> &neuronsA,
                             std::vector<arma::Mat<float>> &weightChanges,
                             std::vector<arma::Col<float>> &biasChanges,
                             Trainer *trainer);

  void train();
  void evalNetwork(bool showPictures);

}; // class NeuralNet

}; // namespace NN
