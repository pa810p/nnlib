#include <gtest/gtest.h>

#include "neuron.h"

TEST(NeuronTest, CalculatesOutputFromInputs) {
    Neuron neuron;
    ASSERT_EQ(neuron.Initialize(2, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);

    double inputs[] = {2.0, -1.0};
    EXPECT_DOUBLE_EQ(neuron.Train(inputs), 0.5);
    EXPECT_DOUBLE_EQ(neuron.getOutput(), 0.5);
    EXPECT_DOUBLE_EQ(neuron.getWeight(0), 0.5);
    EXPECT_DOUBLE_EQ(neuron.getWeight(1), 0.5);
}

TEST(NeuronTest, IncludesConstantInputInOutput) {
    Neuron neuron;
    ASSERT_EQ(neuron.Initialize(2, 0.25, 0.25, ACT_LINEAR, 1.0, 0.1), 0);

    double inputs[] = {2.0, 4.0};
    EXPECT_DOUBLE_EQ(neuron.Train(inputs), 1.75);
    EXPECT_DOUBLE_EQ(neuron.getWeight(2), 0.25);
}

TEST(NeuronTest, CalculatesDeltaAndUpdatesWeights) {
    Neuron neuron;
    ASSERT_EQ(neuron.Initialize(1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);

    double input[] = {2.0};
    ASSERT_DOUBLE_EQ(neuron.Train(input), 1.0);
    EXPECT_DOUBLE_EQ(neuron.countDelta(0.4), 0.4);

    neuron.Modify();

    EXPECT_DOUBLE_EQ(neuron.getWeight(0), 0.58);
    EXPECT_DOUBLE_EQ(neuron.Train(input), 1.16);
}

TEST(NeuronTest, SupportsCustomActivationFunctions) {
    Neuron neuron;
    ASSERT_EQ(neuron.Initialize(1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);

    EXPECT_EQ(neuron.setActivation(act_sigmoid, act_sigmoid_prim), 0);
    EXPECT_EQ(neuron.setActivation(nullptr, act_sigmoid_prim), -1);
    EXPECT_EQ(neuron.setActivation(act_sigmoid, nullptr), -1);
}

TEST(NeuronTest, RejectsUnknownActivationDuringInitialization) {
    Neuron neuron;

    EXPECT_EQ(neuron.Initialize(1, 0.5, 0.5, 999, 0.0, 0.1), -1);
}

TEST(NeuronTest, RejectsUnknownActivationWhenSettingActivation) {
    Neuron neuron;
    ASSERT_EQ(neuron.Initialize(1, 0.5, 0.5, ACT_LINEAR, 0.0, 0.1), 0);

    EXPECT_EQ(neuron.setActivation(999), -1);
}
