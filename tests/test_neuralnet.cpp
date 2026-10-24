#include <gtest/gtest.h>

#include <stdexcept>

#include "neuralnet.h"

TEST(NeuralNetApiTest, RejectsInvalidConstructorArguments) {
    int valid_size[] = {1};
    int invalid_size[] = {0};
    int negative_size[] = {-1};

    EXPECT_THROW(NeuralNet(0, valid_size, 1), std::invalid_argument);
    EXPECT_THROW(NeuralNet(1, nullptr, 1), std::invalid_argument);
    EXPECT_THROW(NeuralNet(1, valid_size, 0), std::invalid_argument);
    EXPECT_THROW(NeuralNet(1, invalid_size, 1), std::invalid_argument);
    EXPECT_THROW(NeuralNet(1, negative_size, 1), std::invalid_argument);
}

TEST(NeuralNetApiTest, ValidatesActivationLayerAndFunction) {
    int layer_sizes[] = {1};
    NeuralNet nn(1, layer_sizes, 2, 0.5, 0.5, 0.2);

    EXPECT_EQ(nn.setActivation(0, ACT_LINEAR), 0);
    EXPECT_EQ(nn.setActivation(-1, ACT_LINEAR), -1);
    EXPECT_EQ(nn.setActivation(1, ACT_LINEAR), -1);
    EXPECT_EQ(nn.setActivation(0, 999), -1);
    EXPECT_EQ(nn.setActivation(0, nullptr, act_linear_prim), -1);
}

TEST(NeuralNetApiTest, ChecksInputAndReturnsNetworkOutput) {
    int layer_sizes[] = {1};
    NeuralNet nn(1, layer_sizes, 2, 0.5, 0.5, 0.2);
    ASSERT_EQ(nn.setActivation(0, ACT_LINEAR), 0);

    double inputs[] = {2.0, -1.0};
    double outputs[1] = {};

    EXPECT_EQ(nn.Check(inputs, outputs), outputs);
    EXPECT_DOUBLE_EQ(outputs[0], 1.0);
}
