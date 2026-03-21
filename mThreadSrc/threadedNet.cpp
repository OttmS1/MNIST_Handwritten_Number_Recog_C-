#include "threadedNet.h"

NN::Trainer::Trainer(NeuralNet *thisNet,
                     std::vector<std::vector<ILPair>> *_batches,
                     const size_t _batchSize)
    : net(thisNet), batches(_batches), goal(numOutputs, arma::fill::zeros),
      batchSize(_batchSize), state(IDLE) 
{
  net->constructLocalNeurons(neuronsZ, neuronsA);
  net->constructLocalChanges(weightChanges, biasChanges);

  thisThread = std::thread([this]() { this->listener(); });
}

NN::Trainer::~Trainer() {
  state.store(STOP);
  state.notify_all();
  if (thisThread.joinable()) {
    thisThread.join();
  }
}

void NN::Trainer::listener() {
  while (this->state.load() != STOP) {
    if (this->state.load() == RUN) {
      startTraining();
      this->state.store(IDLE);
      state.notify_all();
    } else {
      this->state.wait(IDLE);
    }
  }
}

void NN::Trainer::startTraining() {
  for (int j = 0; j < batchSize; j++) {
    net->placeImageInNet((*batches)[batchLocation][j].second, neuronsZ,
                         neuronsA);
    net->feedForward(neuronsZ, neuronsA);

    goal[(*batches)[batchLocation][j].first] = 1.0f;
    net->backprop(goal, neuronsA, weightChanges, biasChanges, this);
    goal.zeros();
  }
}

void NN::NeuralNet::initNetwork() {
  neuronVecDims[0] = numInputs;
  neuronVecDims[LAYERS - 1] = numOutputs;

  for (size_t i = 1; i < LAYERS - 1; i++) {
    neuronVecDims[i] = HIDDEN_LAYER_DIM;
  }

  for (size_t i = 0; i < LAYERS - 1; i++) {
    // He Initialization
    weights[i] = arma::Mat<float>(neuronVecDims[i + 1], neuronVecDims[i],
                                  arma::fill::randn) *
                 std::sqrt(2.0f / neuronVecDims[i]);

    biases[i] = arma::Col<float>(neuronVecDims[i + 1], arma::fill::zeros);

    weightChanges[i] = arma::Mat<float>(neuronVecDims[i + 1], neuronVecDims[i],
                                        arma::fill::zeros);
    biasChanges[i] = arma::Col<float>(neuronVecDims[i + 1], arma::fill::zeros);
  }

  constructLocalNeurons(neuronsZ, neuronsA);
}

void NN::NeuralNet::constructLocalNeurons(
    std::vector<arma::Col<float>> &neuronsZ,
    std::vector<arma::Col<float>> &neuronsA) {
  for (size_t i = 0; i < LAYERS; i++) {
    neuronsA.emplace_back(
        arma::Col<float>(neuronVecDims[i], arma::fill::randu));
    neuronsZ.emplace_back(
        arma::Col<float>(neuronVecDims[i], arma::fill::randu));
  }
}

void NN::NeuralNet::constructLocalChanges(
    std::vector<arma::Mat<float>> &weightChanges,
    std::vector<arma::Col<float>> &biasChanges) {

  for (size_t i = 0; i < LAYERS - 1; i++) {
    weightChanges.emplace_back(arma::Mat<float>(
        neuronVecDims[i + 1], neuronVecDims[i], arma::fill::zeros));

    biasChanges.emplace_back(
        arma::Col<float>(neuronVecDims[i + 1], arma::fill::zeros));
  }
}

NN::NeuralNet::NeuralNet(const char **files)
    : weights(LAYERS - 1), biases(LAYERS - 1), weightChanges(LAYERS - 1),
      biasChanges(LAYERS - 1) {
  std::vector<uint8_t> trainImgsRaw;
  std::vector<uint8_t> testImgsRaw;
  std::vector<std::vector<uint8_t>> trainImgs;
  std::vector<std::vector<uint8_t>> testImgs;
  std::vector<uint8_t> trainLabels;
  std::vector<uint8_t> testLabels;

  MnistParser trainParser(files[0], files[1], trainImgsRaw, trainLabels);
  MnistParser testParser(files[2], files[3], testImgsRaw, testLabels);

  // Seperation of images
  trainImgs =
      std::vector<std::vector<uint8_t>>(trainImgsRaw.size() / numInputs);
  testImgs = std::vector<std::vector<uint8_t>>(testImgsRaw.size() / numInputs);

  for (int i = 0; i < (trainImgsRaw.size()) / numInputs; i++) {
    trainImgs[i].resize(numInputs);

    std::ranges::copy(trainImgsRaw.begin() + i * numInputs,
                      trainImgsRaw.begin() + (i + 1) * numInputs,
                      trainImgs[i].begin());
  }

  for (int i = 0; i < (testImgsRaw.size()) / numInputs; i++) {
    testImgs[i].resize(numInputs);

    std::ranges::copy(testImgsRaw.begin() + i * numInputs,
                      testImgsRaw.begin() + (i + 1) * numInputs,
                      testImgs[i].begin());
  }

  // assign each img to its label
  for (int i = 0; i < trainLabels.size(); i++) {
    this->train_imgLabelPairs.emplace_back(
        std::make_pair(trainLabels[i], trainImgs[i]));
  }

  for (int i = 0; i < testLabels.size(); i++) {
    this->test_imgLabelPairs.emplace_back(
        std::make_pair(testLabels[i], testImgs[i]));
  }

  auto rng = std::default_random_engine{};
  std::ranges::shuffle(this->train_imgLabelPairs, rng);
  std::ranges::shuffle(this->test_imgLabelPairs, rng);

  initNetwork();
}

void NN::NeuralNet::placeImageInNet(std::vector<uint8_t> &img,
                                    std::vector<arma::Col<float>> &neuronsZ,
                                    std::vector<arma::Col<float>> &neuronsA) {
  neuronsZ[0] = arma::conv_to<arma::Mat<float>>::from(img);
  neuronsZ[0] =
      neuronsZ[0] % arma::Col<float>(numInputs, arma::fill::value(1.0f / 255));
  neuronsA[0] = neuronsZ[0];
}
void NN::NeuralNet::feedForward(std::vector<arma::Col<float>> &neuronsZ,
                                std::vector<arma::Col<float>> &neuronsA) {

  for (size_t i = 0; i < LAYERS - 1; i++) {
    neuronsZ[i + 1] = (weights[i] * neuronsA[i]) + biases[i];
    neuronsA[i + 1] = neuronsZ[i + 1];

    if (i == LAYERS - 2) {
      float max_val = neuronsZ[i + 1].max();
      neuronsA[i + 1] = arma::exp(neuronsZ[i + 1] - max_val);
      neuronsA[i + 1] /= arma::accu(neuronsA[i + 1]);
    } else {
      neuronsA[i + 1].transform([](float val) { return std::max(0.0f, val); });
    }
  }
}

arma::Col<float>
NN::NeuralNet::layerGrad(const arma::Col<float> &errorL, const size_t layerNum,
                         std::vector<arma::Col<float>> &neuronsA,
                         std::vector<arma::Mat<float>> &weightChanges,
                         std::vector<arma::Col<float>> &biasChanges,
                         Trainer *trainer) {

  arma::Col<float> currLayerError;
  if (layerNum == LAYERS - 1) {
    currLayerError = errorL;
  } else {
    arma::Col<float> deriv =
        arma::conv_to<arma::Col<float>>::from(neuronsA[layerNum] > 0);
    currLayerError = errorL % deriv;
  }

  (trainer->weightChanges)[layerNum - 1] +=
      currLayerError * neuronsA[layerNum - 1].t();
  (trainer->biasChanges)[layerNum - 1] += currLayerError;

  return weights[layerNum - 1].t() * currLayerError;
}

arma::Col<float> &NN::NeuralNet::backprop(
    arma::Col<float> &goal, std::vector<arma::Col<float>> &neuronsA,
    std::vector<arma::Mat<float>> &weightChanges,
    std::vector<arma::Col<float>> &biasChanges, Trainer *trainer) {
  arma::Col<float> delta = neuronsA[LAYERS - 1] - goal;

  for (size_t i = LAYERS - 1; i > 0; i--) {
    delta = layerGrad(delta, i, neuronsA, weightChanges, biasChanges, trainer);
  }

  return neuronsA[LAYERS - 1];
}

void NN::NeuralNet::train() {
  const size_t TRAINING_STEPS = 10000;
  constexpr int numThreads = 16;
  constexpr int batchSize = 32;
  const int numBatches = train_imgLabelPairs.size() / batchSize;

  std::vector<std::vector<ILPair>> batches(numBatches);
  for (int i = 0; i < train_imgLabelPairs.size() / batchSize; i++) {
    batches[i] = std::vector<ILPair>(batchSize);
    std::ranges::copy(train_imgLabelPairs.begin() + i * batchSize,
                      train_imgLabelPairs.begin() + (i + 1) * batchSize,
                      batches[i].begin());
  }

  std::vector<std::unique_ptr<Trainer>> trainers;
  for (int i = 0; i < numThreads; i++) {
    trainers.push_back(std::make_unique<Trainer>(this, &batches, batchSize));
  }

  for (size_t epoch = 0; epoch < TRAINING_STEPS; epoch++) {

    for (std::unique_ptr<Trainer> &trainer : trainers) {
      trainer->batchLocation = rand() % numBatches;
      trainer->state.store(RUN);
      trainer->state.notify_all();
    }

    for (std::unique_ptr<Trainer> &trainer : trainers) {
      trainer->state.wait(RUN);
      trainer->state.notify_all();
    }

    for (std::unique_ptr<Trainer> &trainer : trainers) {
      for (size_t i = 0; i < LAYERS - 1; i++) {
        weights[i] -=
            (trainer->weightChanges)[i] * ETA / (batchSize * numThreads);
        biases[i] -= (trainer->biasChanges)[i] * ETA / (batchSize * numThreads);

        trainer->resetWeightChanges();
        trainer->resetBiasChanges();
      }
    }
    if (epoch % 1000 == 0) {
      std::printf("%s", "Epoch ");
      std::printf("%i", epoch);
      std::printf("%s", ":\t");
      this->evalNetwork(false);
    }
  }
  this->evalNetwork(true);
}

void NN::NeuralNet::evalNetwork(bool showPictures) {
  double diff = 0;
  arma::Col<float> goal(10, arma::fill::zeros);

  if (!showPictures) {
    size_t i = 0;
    for (; i < 10000; i++) {
      this->placeImageInNet((this->test_imgLabelPairs[i]).second, neuronsZ,
                            neuronsA);
      this->feedForward(neuronsZ, neuronsA);

      goal[(this->test_imgLabelPairs[i]).first] = 1;

      double cost = arma::accu((this->neuronsA[LAYERS - 1] - goal) %
                               (this->neuronsA[LAYERS - 1] - goal));

      diff += cost;
      goal.zeros();
    }

    std::printf("%s", "average cost across test examples: ");
    std::printf("%f", diff / i);
    std::printf("%c", '\n');

  } else {
    size_t i = 0;
    for (; i < 10000; i++) {
      this->placeImageInNet((this->test_imgLabelPairs[i]).second, neuronsZ,
                            neuronsA);
      this->feedForward(neuronsZ, neuronsA);

      goal[(this->test_imgLabelPairs[i]).first] = 1.0f;

      double cost = arma::accu((this->neuronsA[LAYERS - 1] - goal) %
                               (this->neuronsA[LAYERS - 1] - goal));
      std::printf("%s",
                  "------------------------------------------------------\n");
      for (size_t z = 0; z < test_imgLabelPairs[i].second.size(); z++) {
        if (test_imgLabelPairs[i].second[z] < 0.3) {
          std::printf("%s", ". ");
        } else {
          std::printf("%s", "# ");
        }

        if (z % 28 == 27) {
          std::printf("%c", '\n');
        }
      }
      std::printf("%s",
                  "------------------------------------------------------\n");

      std::printf("%s", "Goal: ");
      std::printf("%u", test_imgLabelPairs[i].first);
      std::printf("%c", '\n');

      std::printf("%s", "Guess: ");
      std::printf("%u", neuronsA[LAYERS - 1].index_max());
      std::printf("%c", '\t');
      std::printf("%s", "Confidence: ");
      std::printf("%f", neuronsA[LAYERS - 1].max());
      std::printf("%c", '\n');
      std::printf("%c", '\n');

      diff += cost;
      goal.zeros();
    }

    std::printf("%s", "average cost across test examples: ");
    std::printf("%f", diff / i);
    std::printf("%c", '\n');
  }
}
