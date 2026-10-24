#include <gtest/gtest.h>

#include <cstdlib>
#include <span>
#include <vector>

#include "neuralnet.h"

namespace {

double rand_range(double lo, double hi) {
    return lo + (hi - lo) * (static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0));
}

double evaluate_mse(NeuralNet& nn, std::span<const double> inputs,
                    std::span<const double> targets,
                    int samples, int input_size, int output_size) {
    std::vector<double> predictions(static_cast<std::size_t>(output_size));
    double sum = 0.0;

    for (int i = 0; i < samples; ++i) {
        const auto input_offset = static_cast<std::size_t>(i * input_size);
        const auto output_offset = static_cast<std::size_t>(i * output_size);
        nn.Check(inputs.subspan(input_offset, static_cast<std::size_t>(input_size)),
                 predictions);
        for (int j = 0; j < output_size; ++j) {
            const double error = targets[output_offset + static_cast<std::size_t>(j)] -
                                 predictions[static_cast<std::size_t>(j)];
            sum += error * error;
        }
    }

    return sum / (samples * output_size);
}

}

TEST(NeuralNetScenarioTest, LearnsOrDataset) {
    int layer_sizes[] = {3, 1};
    constexpr int input_size = 2;
    constexpr int samples = 4;
    double inputs[samples * input_size] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    double targets[samples] = {0.0, 1.0, 1.0, 1.0};

    srand(7);
    NeuralNet nn(layer_sizes, input_size, -0.25, 0.25, 0.6);
    nn.Train(samples, inputs, targets, 1e-5, 60000);

    EXPECT_LT(evaluate_mse(nn, inputs, targets, samples, input_size, 1), 0.03);
}

TEST(NeuralNetScenarioTest, LearnsLinearRegressionDataset) {
    int layer_sizes[] = {1};
    constexpr int input_size = 2;
    constexpr int samples = 8;
    double inputs[samples * input_size] = {
        -1.0, -1.0,
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0,
         0.5, -0.5,
        -0.5,  0.5,
         2.0,  0.0,
         0.0,  2.0
    };
    double targets[samples] = {
        -3.0, 1.0, -1.0, 3.0, -0.5, 0.5, 2.0, 4.0
    };

    srand(11);
    NeuralNet nn(layer_sizes, input_size, -0.25, 0.25, 0.2);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 30000);

    EXPECT_LT(evaluate_mse(nn, inputs, targets, samples, input_size, 1), 1e-4);
}

TEST(NeuralNetScenarioTest, LearnsLinearIdentityDataset) {
    int layer_sizes[] = {2};
    constexpr int input_size = 2;
    constexpr int output_size = 2;
    constexpr int samples = 6;
    double inputs[samples * input_size] = {
        -1.0, -2.0,
         0.0,  0.0,
         1.0,  2.0,
         2.5, -1.5,
        -3.0,  1.0,
         0.5,  0.25
    };
    double targets[samples * output_size] = {
        -1.0, -2.0,
         0.0,  0.0,
         1.0,  2.0,
         2.5, -1.5,
        -3.0,  1.0,
         0.5,  0.25
    };

    srand(19);
    NeuralNet nn(layer_sizes, input_size, -0.25, 0.25, 0.15);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 40000);

    EXPECT_LT(evaluate_mse(nn, inputs, targets, samples, input_size, output_size), 1e-4);
}

TEST(NeuralNetScenarioTest, LearnsRandomLinearSystemWithSingleOutput) {
    int layer_sizes[] = {1};
    constexpr int input_size = 3;
    constexpr int samples = 96;
    double inputs[samples * input_size];
    double targets[samples];
    constexpr double w1 = 1.2;
    constexpr double w2 = -0.7;
    constexpr double w3 = 0.3;
    constexpr double bias = -0.4;

    srand(29);
    for (int i = 0; i < samples; ++i) {
        const double x1 = rand_range(-2.0, 2.0);
        const double x2 = rand_range(-2.0, 2.0);
        const double x3 = rand_range(-2.0, 2.0);
        inputs[i * input_size] = x1;
        inputs[i * input_size + 1] = x2;
        inputs[i * input_size + 2] = x3;
        targets[i] = w1 * x1 + w2 * x2 + w3 * x3 + bias;
    }

    srand(31);
    NeuralNet nn(layer_sizes, input_size, -0.25, 0.25, 0.08);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 60000);

    EXPECT_LT(evaluate_mse(nn, inputs, targets, samples, input_size, 1), 2e-4);
}

TEST(NeuralNetScenarioTest, LearnsRandomLinearSystemWithTwoOutputs) {
    int layer_sizes[] = {2};
    constexpr int input_size = 3;
    constexpr int output_size = 2;
    constexpr int samples = 128;
    double inputs[samples * input_size];
    double targets[samples * output_size];

    srand(37);
    for (int i = 0; i < samples; ++i) {
        const double x1 = rand_range(-3.0, 3.0);
        const double x2 = rand_range(-3.0, 3.0);
        const double x3 = rand_range(-3.0, 3.0);
        inputs[i * input_size] = x1;
        inputs[i * input_size + 1] = x2;
        inputs[i * input_size + 2] = x3;
        targets[i * output_size] = 0.5 * x1 - 1.3 * x2 + 0.9 * x3 + 0.2;
        targets[i * output_size + 1] = -0.8 * x1 + 0.4 * x2 + 0.6 * x3 - 1.1;
    }

    srand(41);
    NeuralNet nn(layer_sizes, input_size, -0.25, 0.25, 0.05);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 90000);

    EXPECT_LT(evaluate_mse(nn, inputs, targets, samples, input_size, output_size), 3e-4);
}
