#include <gtest/gtest.h>

#include <span>
#include <stdexcept>

#include "layer.h"

TEST(NeuralLayerTest, ForwardsInputsThroughEveryNeuron) {
    NeuralLayer layer;
    ASSERT_EQ(layer.Initialize(2, 2, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);
    EXPECT_EQ(layer.getLayerSize(), 2);

    double inputs[] = {2.0, -1.0};
    ASSERT_EQ(layer.Train(inputs), 0);

    EXPECT_DOUBLE_EQ(layer.getOutput(0), 0.5);
    EXPECT_DOUBLE_EQ(layer.getOutput(1), 0.5);
}

TEST(NeuralLayerTest, TrainsFromPreviousLayerOutputs) {
    NeuralLayer previous;
    NeuralLayer next;
    ASSERT_EQ(previous.Initialize(1, 1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);
    ASSERT_EQ(next.Initialize(1, 1, 0.25, 0.25, ACT_LINEAR, 0.0, 0.1), 0);

    double input[] = {2.0};
    ASSERT_EQ(previous.Train(input), 0);
    ASSERT_EQ(next.Train(previous), 0);

    EXPECT_DOUBLE_EQ(previous.getOutput(0), 1.0);
    EXPECT_DOUBLE_EQ(next.getOutput(0), 0.25);
}

TEST(NeuralLayerTest, CalculatesOutputErrorsAndDeltas) {
    NeuralLayer layer;
    ASSERT_EQ(layer.Initialize(2, 1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);

    double input[] = {2.0};
    ASSERT_EQ(layer.Train(input), 0);

    double desired[] = {1.0, -1.0};
    double errors[2] = {};
    layer.countLastDelta(desired, errors);

    EXPECT_DOUBLE_EQ(errors[0], 0.0);
    EXPECT_DOUBLE_EQ(errors[1], -2.0);
    EXPECT_DOUBLE_EQ(layer.getNeuronDelta(0), 0.0);
    EXPECT_DOUBLE_EQ(layer.getNeuronDelta(1), -2.0);
}

TEST(NeuralLayerTest, PropagatesWeightedDeltaToPreviousLayer) {
    NeuralLayer previous;
    NeuralLayer next;
    ASSERT_EQ(previous.Initialize(1, 1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);
    ASSERT_EQ(next.Initialize(1, 1, 0.25, 0.25, ACT_LINEAR, 0.0, 0.1), 0);

    double input[] = {2.0};
    ASSERT_EQ(previous.Train(input), 0);
    ASSERT_EQ(next.Train(previous), 0);

    double desired[] = {1.5};
    double errors[1] = {};
    next.countLastDelta(desired, errors);
    previous.countDelta(next);

    EXPECT_DOUBLE_EQ(errors[0], 1.25);
    EXPECT_DOUBLE_EQ(next.getNeuronDelta(0), 1.25);
    EXPECT_DOUBLE_EQ(previous.getNeuronDelta(0), 0.3125);
}

TEST(NeuralLayerTest, AppliesActivationToAllNeurons) {
    NeuralLayer layer;
    ASSERT_EQ(layer.Initialize(1, 1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);

    EXPECT_EQ(layer.setActivation(ACT_SIGMOID), 0);
    EXPECT_EQ(layer.setActivation(999), -1);
    EXPECT_EQ(layer.setActivation(act_linear, nullptr), -1);
}

TEST(NeuralLayerTest, RejectsNonPositiveLayerOrInputSizes) {
    NeuralLayer zero_neurons;
    NeuralLayer negative_neurons;
    NeuralLayer zero_inputs;
    NeuralLayer negative_inputs;

    EXPECT_EQ(zero_neurons.Initialize(0, 1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), -1);
    EXPECT_EQ(negative_neurons.Initialize(-1, 1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), -1);
    EXPECT_EQ(zero_inputs.Initialize(1, 0, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), -1);
    EXPECT_EQ(negative_inputs.Initialize(1, -1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), -1);
}

TEST(NeuralLayerTest, PropagatesNeuronInitializationFailure) {
    NeuralLayer layer;

    EXPECT_EQ(layer.Initialize(1, 1, 0.5, 0.5, 999, 0.0, 0.1), -1);
}

TEST(NeuralLayerTest, RejectsInputAndOutputSpansWithIncorrectSizes) {
    NeuralLayer layer;
    NeuralLayer previous;
    NeuralLayer mismatched_next;
    ASSERT_EQ(layer.Initialize(1, 2, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);
    ASSERT_EQ(previous.Initialize(1, 1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);
    ASSERT_EQ(mismatched_next.Initialize(1, 2, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);
    double short_input[] = {1.0};
    std::span<const double> short_desired;
    std::span<double> short_error;

    EXPECT_THROW(layer.Train(std::span<const double>(short_input)), std::invalid_argument);
    EXPECT_THROW(layer.Train(previous), std::invalid_argument);
    EXPECT_THROW(layer.countLastDelta(short_desired, short_error), std::invalid_argument);
    EXPECT_THROW(layer.countDelta(mismatched_next), std::invalid_argument);
}
