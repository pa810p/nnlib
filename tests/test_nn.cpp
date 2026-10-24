#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "neuralnet.h"

static double rand_range(double lo, double hi) {
    return lo + (hi - lo) * ((double)rand() / ((double)RAND_MAX + 1.0));
}

static double evaluate_mse(NeuralNet &nn, const double *inputs, const double *targets, int samples, int input_size, int output_size) {
    double *pred = new double[output_size];
    double sum = 0.0;

    for (int i = 0; i < samples; ++i) {
        nn.Check((double *)(inputs + i * input_size), pred);
        for (int j = 0; j < output_size; ++j) {
            double e = targets[i * output_size + j] - pred[j];
            sum += e * e;
        }
    }

    delete[] pred;
    return sum / (samples * output_size);
}

static int test_or_dataset() {
    // OR gate with one hidden layer and linear output.
    const int layer_count = 2;
    int ls[layer_count] = {3, 1};
    const int input_size = 2;
    const int samples = 4;

    double inputs[samples * input_size] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };

    double targets[samples] = {0.0, 1.0, 1.0, 1.0};

    srand(7);
    NeuralNet nn(layer_count, ls, input_size, -0.25, 0.25, 0.6);
    nn.Train(samples, inputs, targets, 1e-5, 60000);

    double mse = evaluate_mse(nn, inputs, targets, samples, input_size, 1);
    printf("[test_or_dataset] mse=%f\n", mse);

    return (mse < 0.03) ? 0 : 1;
}

static int test_linear_regression_dataset() {
    // Pure linear mapping y = x1 + 2*x2 with one linear layer.
    const int layer_count = 1;
    int ls[layer_count] = {1};
    const int input_size = 2;
    const int samples = 8;

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
        -3.0,
         1.0,
        -1.0,
         3.0,
        -0.5,
         0.5,
         2.0,
         4.0
    };

    srand(11);
    NeuralNet nn(layer_count, ls, input_size, -0.25, 0.25, 0.2);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 30000);

    double mse = evaluate_mse(nn, inputs, targets, samples, input_size, 1);
    printf("[test_linear_regression_dataset] mse=%f\n", mse);

    return (mse < 1e-4) ? 0 : 1;
}

static int test_linear_identity_dataset() {
    // Two-output linear identity mapping [x1, x2] -> [x1, x2].
    const int layer_count = 1;
    int ls[layer_count] = {2};
    const int input_size = 2;
    const int output_size = 2;
    const int samples = 6;

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
    NeuralNet nn(layer_count, ls, input_size, -0.25, 0.25, 0.15);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 40000);

    double mse = evaluate_mse(nn, inputs, targets, samples, input_size, output_size);
    printf("[test_linear_identity_dataset] mse=%f\n", mse);

    return (mse < 1e-4) ? 0 : 1;
}

static int test_set_activation_bounds() {
    const int layer_count = 1;
    int ls[layer_count] = {1};
    const int input_size = 2;

    srand(23);
    NeuralNet nn(layer_count, ls, input_size, -0.25, 0.25, 0.2);

    int ok_valid = nn.setActivation(0, ACT_LINEAR);
    int err_neg = nn.setActivation(-1, ACT_LINEAR);
    int err_hi = nn.setActivation(layer_count, ACT_LINEAR);
    int err_activation = nn.setActivation(0, 999);

    printf("[test_set_activation_bounds] valid=%d neg=%d hi=%d activation=%d\n", ok_valid, err_neg, err_hi, err_activation);

    return (ok_valid == 0 && err_neg == -1 && err_hi == -1 && err_activation == -1) ? 0 : 1;
}

static int test_constructor_validation() {
    int valid_size[] = {1};
    bool zero_layers_rejected = false;
    bool zero_inputs_rejected = false;
    bool zero_size_rejected = false;

    try { NeuralNet nn(0, valid_size, 1); } catch (...) { zero_layers_rejected = true; }
    try { NeuralNet nn(1, valid_size, 0); } catch (...) { zero_inputs_rejected = true; }
    int invalid_size[] = {0};
    try { NeuralNet nn(1, invalid_size, 1); } catch (...) { zero_size_rejected = true; }

    printf("[test_constructor_validation] layers=%d inputs=%d size=%d\n", zero_layers_rejected, zero_inputs_rejected, zero_size_rejected);
    return (zero_layers_rejected && zero_inputs_rejected && zero_size_rejected) ? 0 : 1;
}

static int test_jump_derivative() {
    double negative = act_jump_prim(-1.0);
    double zero = act_jump_prim(0.0);
    double positive = act_jump_prim(1.0);

    printf("[test_jump_derivative] negative=%f zero=%f positive=%f\n", negative, zero, positive);
    return (negative == 0.0 && zero == 0.0 && positive == 0.0) ? 0 : 1;
}

static int test_random_linear_system_single_output() {
    // Stress linear fitting on synthetic random data with known target linear model.
    const int layer_count = 1;
    int ls[layer_count] = {1};
    const int input_size = 3;
    const int samples = 96;

    double inputs[samples * input_size];
    double targets[samples];

    const double w1 = 1.2;
    const double w2 = -0.7;
    const double w3 = 0.3;
    const double b = -0.4;

    srand(29);
    for (int i = 0; i < samples; ++i) {
        double x1 = rand_range(-2.0, 2.0);
        double x2 = rand_range(-2.0, 2.0);
        double x3 = rand_range(-2.0, 2.0);

        inputs[i * input_size + 0] = x1;
        inputs[i * input_size + 1] = x2;
        inputs[i * input_size + 2] = x3;
        targets[i] = w1 * x1 + w2 * x2 + w3 * x3 + b;
    }

    srand(31);
    NeuralNet nn(layer_count, ls, input_size, -0.25, 0.25, 0.08);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 60000);

    double mse = evaluate_mse(nn, inputs, targets, samples, input_size, 1);
    printf("[test_random_linear_system_single_output] mse=%f\n", mse);

    return (mse < 2e-4) ? 0 : 1;
}

static int test_random_linear_system_two_output() {
    const int layer_count = 1;
    int ls[layer_count] = {2};
    const int input_size = 3;
    const int output_size = 2;
    const int samples = 128;

    double inputs[samples * input_size];
    double targets[samples * output_size];

    // y1 = 0.5*x1 - 1.3*x2 + 0.9*x3 + 0.2
    // y2 = -0.8*x1 + 0.4*x2 + 0.6*x3 - 1.1
    srand(37);
    for (int i = 0; i < samples; ++i) {
        double x1 = rand_range(-3.0, 3.0);
        double x2 = rand_range(-3.0, 3.0);
        double x3 = rand_range(-3.0, 3.0);

        inputs[i * input_size + 0] = x1;
        inputs[i * input_size + 1] = x2;
        inputs[i * input_size + 2] = x3;

        targets[i * output_size + 0] = 0.5 * x1 - 1.3 * x2 + 0.9 * x3 + 0.2;
        targets[i * output_size + 1] = -0.8 * x1 + 0.4 * x2 + 0.6 * x3 - 1.1;
    }

    srand(41);
    NeuralNet nn(layer_count, ls, input_size, -0.25, 0.25, 0.05);
    nn.setAllActivation(ACT_LINEAR);
    nn.Train(samples, inputs, targets, 1e-8, 90000);

    double mse = evaluate_mse(nn, inputs, targets, samples, input_size, output_size);
    printf("[test_random_linear_system_two_output] mse=%f\n", mse);

    return (mse < 3e-4) ? 0 : 1;
}

int main() {
    int failed = 0;

    failed += test_or_dataset();
    failed += test_linear_regression_dataset();
    failed += test_linear_identity_dataset();
    failed += test_set_activation_bounds();
    failed += test_constructor_validation();
    failed += test_jump_derivative();
    failed += test_random_linear_system_single_output();
    failed += test_random_linear_system_two_output();

    if (failed == 0) {
        printf("All tests passed.\n");
        return 0;
    }

    printf("%d test(s) failed.\n", failed);
    return 1;
}
