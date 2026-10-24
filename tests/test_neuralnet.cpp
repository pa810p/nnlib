#include <gtest/gtest.h>

#include <span>
#include <stdexcept>

#include "neuralnet.h"

TEST(NeuralNetApiTest, RejectsInvalidConstructorArguments) {
    int valid_size[] = {1};
    int invalid_size[] = {0};
    int negative_size[] = {-1};

    EXPECT_THROW(NeuralNet(std::span<const int>{}, 1), std::invalid_argument);
    EXPECT_THROW(NeuralNet(valid_size, 0), std::invalid_argument);
    EXPECT_THROW(NeuralNet(invalid_size, 1), std::invalid_argument);
    EXPECT_THROW(NeuralNet(negative_size, 1), std::invalid_argument);
}

TEST(NeuralNetApiTest, ValidatesActivationLayerAndFunction) {
    int layer_sizes[] = {1};
    NeuralNet nn(layer_sizes, 2, 0.5, 0.5, 0.2);

    EXPECT_EQ(nn.setActivation(0, ACT_LINEAR), 0);
    EXPECT_EQ(nn.setActivation(-1, ACT_LINEAR), -1);
    EXPECT_EQ(nn.setActivation(1, ACT_LINEAR), -1);
    EXPECT_EQ(nn.setActivation(0, 999), -1);
    EXPECT_EQ(nn.setActivation(0, nullptr, act_linear_prim), -1);
}

TEST(NeuralNetApiTest, ChecksInputAndWritesNetworkOutput) {
    int layer_sizes[] = {1};
    NeuralNet nn(layer_sizes, 2, 0.5, 0.5, 0.2);
    ASSERT_EQ(nn.setActivation(0, ACT_LINEAR), 0);

    double inputs[] = {2.0, -1.0};
    double outputs[1] = {};

    nn.Check(inputs, outputs);
    EXPECT_DOUBLE_EQ(outputs[0], 1.0);
}

TEST(NeuralNetApiTest, RejectsInputAndOutputSpansWithIncorrectSizes) {
    int layer_sizes[] = {1};
    NeuralNet nn(layer_sizes, 2, 0.5, 0.5, 0.2);
    double short_inputs[] = {1.0};
    double inputs[] = {1.0, 2.0};
    std::span<double> short_outputs;

    EXPECT_THROW(nn.Check(short_inputs, short_outputs), std::invalid_argument);
    EXPECT_THROW(nn.Check(inputs, short_outputs), std::invalid_argument);
}

TEST(NeuralNetApiTest, RejectsTrainingBuffersThatDoNotMatchSampleCount) {
    int layer_sizes[] = {1};
    NeuralNet nn(layer_sizes, 2, 0.5, 0.5, 0.2);
    double inputs[] = {1.0, 2.0};
    double outputs[] = {1.0};

    EXPECT_THROW(nn.Train(0, inputs, outputs, 1e-5, 10), std::invalid_argument);
    EXPECT_THROW(nn.Train(2, inputs, outputs, 1e-5, 10), std::invalid_argument);
}
