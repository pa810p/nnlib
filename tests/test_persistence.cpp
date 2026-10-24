#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <nnl/neuralnet.h>
#include "training_data.h"

namespace {

class TemporaryModelFile {
public:
    explicit TemporaryModelFile(const std::string& suffix) {
        const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("nnlib_" + suffix + "_" + std::to_string(id) + ".json");
    }

    ~TemporaryModelFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    std::filesystem::path path;
};

void write_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.is_open());
    file << contents;
    ASSERT_TRUE(file.good());
}

double custom_identity(double value) {
    return value;
}

double custom_identity_derivative(double) {
    return 1.0;
}

void expect_same_predictions(NeuralNet& first, NeuralNet& second,
                             const TrainingData& data) {
    const auto output_size = static_cast<std::size_t>(data.layer_sizes.back());
    std::vector<double> first_output(output_size);
    std::vector<double> second_output(output_size);
    for (const TrainingSample& sample : data.training) {
        first.Check(sample.input, first_output);
        second.Check(sample.input, second_output);
        for (std::size_t output = 0; output < output_size; ++output)
            EXPECT_DOUBLE_EQ(first_output[output], second_output[output]);
    }
    for (const std::vector<double>& check : data.checks) {
        first.Check(check, first_output);
        second.Check(check, second_output);
        for (std::size_t output = 0; output < output_size; ++output)
            EXPECT_DOUBLE_EQ(first_output[output], second_output[output]);
    }
}

std::vector<double> flatten_training_samples(const TrainingData& data,
                                             std::size_t first_sample,
                                             std::size_t sample_count,
                                             bool use_targets) {
    const std::size_t value_count = use_targets
        ? static_cast<std::size_t>(data.layer_sizes.back())
        : data.input_size;
    std::vector<double> values;
    values.reserve(sample_count * value_count);
    for (std::size_t index = first_sample; index < first_sample + sample_count; ++index) {
        const auto& sample_values = use_targets ? data.training[index].target
                                                : data.training[index].input;
        values.insert(values.end(), sample_values.begin(), sample_values.end());
    }
    return values;
}

std::string valid_model_json() {
    return R"({"format":"nnlib-model","version":1,"input_size":1,"layers":[{"input_size":1,"neurons":[{"activation":"linear","weights":[1.0],"bias_input":1.0,"bias_weight":0.5,"learning_rate":0.1,"momentum_coefficient":0.2,"momentum":[0.0],"bias_momentum":0.0}]}]})";
}

}

TEST(PersistenceTest, RoundTripsPredictionsAndBuiltInActivations) {
    TemporaryModelFile file("round_trip");
    constexpr int samples = 4;
    std::array<int, 2> layer_sizes{2, 1};
    std::array<double, samples * 2> inputs{
        -1.0, -1.0,
        -1.0, 1.0,
        1.0, -1.0,
        1.0, 1.0
    };
    std::array<double, samples> outputs{-3.0, 1.0, -1.0, 3.0};

    std::srand(101);
    NeuralNet original(layer_sizes, 2, -0.2, 0.2, 0.1);
    original.setAllActivation(ACT_LINEAR);
    original.setMomentum(0.25);
    original.Train(samples, inputs, outputs, -1.0, 20);

    const std::array<double, 2> sample{0.35, -0.8};
    std::array<double, 1> expected{};
    original.Check(sample, expected);
    original.save(file.path);

    NeuralNet loaded = NeuralNet::load(file.path);
    std::array<double, 1> actual{};
    loaded.Check(sample, actual);

    EXPECT_DOUBLE_EQ(actual[0], expected[0]);
}

TEST(PersistenceTest, RestoresOptimizerStateForContinuedTraining) {
    TemporaryModelFile file("resume_training");
    constexpr int samples = 4;
    std::array<int, 1> layer_sizes{1};
    std::array<double, samples * 2> inputs{
        -1.0, -1.0,
        -1.0, 1.0,
        1.0, -1.0,
        1.0, 1.0
    };
    std::array<double, samples> outputs{-3.0, 1.0, -1.0, 3.0};

    std::srand(103);
    NeuralNet original(layer_sizes, 2, -0.2, 0.2, 0.08);
    original.setAllActivation(ACT_LINEAR);
    original.setMomentum(0.35);
    original.Train(samples, inputs, outputs, -1.0, 30);
    original.save(file.path);

    NeuralNet resumed = NeuralNet::load(file.path);
    original.Train(samples, inputs, outputs, -1.0, 25);
    resumed.Train(samples, inputs, outputs, -1.0, 25);

    std::array<double, 2> sample{-0.45, 0.9};
    std::array<double, 1> original_output{};
    std::array<double, 1> resumed_output{};
    original.Check(sample, original_output);
    resumed.Check(sample, resumed_output);
    EXPECT_DOUBLE_EQ(original_output[0], resumed_output[0]);
}

TEST(PersistenceTest, ResumesOnSecondHalfOfTrains2AndPersistsAgain) {
    TemporaryModelFile file("trains2_resume");
    const TrainingData data = load_training_data(
        std::filesystem::path(NNLIB_TEST_RESOURCE_DIR) / "trains2.json");
    ASSERT_EQ(data.training.size(), 8);
    ASSERT_EQ(data.input_size, 3);
    ASSERT_EQ(data.layer_sizes, (std::vector<int>{6, 4, 2}));

    constexpr std::size_t first_half_count = 4;
    const auto output_size = static_cast<std::size_t>(data.layer_sizes.back());
    const auto second_sample_count = data.training.size() - first_half_count;
    const std::vector<double> first_inputs =
        flatten_training_samples(data, 0, first_half_count, false);
    const std::vector<double> first_targets =
        flatten_training_samples(data, 0, first_half_count, true);
    const std::vector<double> second_inputs =
        flatten_training_samples(data, first_half_count, second_sample_count, false);
    const std::vector<double> second_targets =
        flatten_training_samples(data, first_half_count, second_sample_count, true);

    std::srand(107);
    NeuralNet initial(data.layer_sizes, static_cast<int>(data.input_size), -0.25, 0.25, 0.2);
    initial.setMomentum(0.2);
    initial.Train(static_cast<int>(first_half_count), first_inputs, first_targets,
                  -1.0, 40);
    initial.save(file.path);

    std::srand(107);
    NeuralNet uninterrupted(data.layer_sizes, static_cast<int>(data.input_size),
                            -0.25, 0.25, 0.2);
    uninterrupted.setMomentum(0.2);
    uninterrupted.Train(static_cast<int>(first_half_count), first_inputs, first_targets,
                        -1.0, 40);
    uninterrupted.Train(static_cast<int>(second_sample_count), second_inputs,
                        second_targets, -1.0, 40);

    NeuralNet resumed = NeuralNet::load(file.path);
    resumed.Train(static_cast<int>(second_sample_count), second_inputs,
                  second_targets, -1.0, 40);
    resumed.save(file.path);

    NeuralNet loaded_again = NeuralNet::load(file.path);
    expect_same_predictions(uninterrupted, loaded_again, data);
}

TEST(PersistenceTest, RestoresRegisteredCustomActivation) {
    TemporaryModelFile file("custom_activation");
    registerActivation("custom.test.identity", custom_identity,
                      custom_identity_derivative);
    std::array<int, 1> layer_sizes{1};
    std::array<double, 1> inputs{0.75};
    std::array<double, 1> original_output{};
    std::array<double, 1> loaded_output{};

    NeuralNet original(layer_sizes, 1, 0.4, 0.4, 0.1);
    ASSERT_EQ(original.setActivation(0, "custom.test.identity"), 0);
    original.Check(inputs, original_output);
    original.save(file.path);

    NeuralNet loaded = NeuralNet::load(file.path);
    loaded.Check(inputs, loaded_output);

    EXPECT_DOUBLE_EQ(loaded_output[0], original_output[0]);
}

TEST(PersistenceTest, RejectsUnregisteredCustomActivationOnSave) {
    TemporaryModelFile file("unregistered_activation");
    std::array<int, 1> layer_sizes{1};
    NeuralNet network(layer_sizes, 1, 0.4, 0.4, 0.1);

    ASSERT_EQ(network.setActivation(0, custom_identity, custom_identity_derivative), 0);
    EXPECT_THROW(network.save(file.path), std::runtime_error);
}

TEST(PersistenceTest, RejectsDuplicateAndInvalidActivationRegistrations) {
    registerActivation("custom.test.duplicate", custom_identity,
                      custom_identity_derivative);
    EXPECT_THROW(registerActivation("custom.test.duplicate", custom_identity,
                                    custom_identity_derivative), std::invalid_argument);
    EXPECT_THROW(registerActivation("sigmoid", custom_identity,
                                    custom_identity_derivative), std::invalid_argument);
    EXPECT_THROW(registerActivation("custom.test.null", nullptr,
                                    custom_identity_derivative), std::invalid_argument);
}

TEST(PersistenceTest, RejectsMalformedAndUnsupportedModelFiles) {
    TemporaryModelFile file("invalid_model");

    write_file(file.path, "{not-json");
    EXPECT_THROW(NeuralNet::load(file.path), std::runtime_error);

    write_file(file.path, R"({"format":"nnlib-model","version":99,"input_size":1,"layers":[]})");
    EXPECT_THROW(NeuralNet::load(file.path), std::runtime_error);

    write_file(file.path, R"({"input_size":1,"layers":[]})");
    EXPECT_THROW(NeuralNet::load(file.path), std::runtime_error);

    write_file(file.path, valid_model_json());
    NeuralNet valid = NeuralNet::load(file.path);
    SUCCEED();
}

TEST(PersistenceTest, RejectsInvalidDimensionsAndUnknownActivations) {
    TemporaryModelFile file("invalid_dimensions");

    write_file(file.path,
        R"({"format":"nnlib-model","version":1,"input_size":1,"layers":[{"input_size":1,"neurons":[{"activation":"linear","weights":[],"bias_input":1.0,"bias_weight":0.5,"learning_rate":0.1,"momentum_coefficient":0.0,"momentum":[0.0],"bias_momentum":0.0}]}]})");
    EXPECT_THROW(NeuralNet::load(file.path), std::runtime_error);

    write_file(file.path,
        R"({"format":"nnlib-model","version":1,"input_size":1,"layers":[{"input_size":1,"neurons":[{"activation":"custom.missing","weights":[1.0],"bias_input":1.0,"bias_weight":0.5,"learning_rate":0.1,"momentum_coefficient":0.0,"momentum":[0.0],"bias_momentum":0.0}]}]})");
    EXPECT_THROW(NeuralNet::load(file.path), std::runtime_error);
}

TEST(PersistenceTest, ReportsFileErrors) {
    const auto missing_path = std::filesystem::temp_directory_path() /
                              "nnlib_missing_model_file.json";
    std::error_code error;
    std::filesystem::remove(missing_path, error);
    EXPECT_THROW(NeuralNet::load(missing_path), std::runtime_error);

    std::array<int, 1> layer_sizes{1};
    NeuralNet network(layer_sizes, 1, 0.0, 0.0, 0.1);
    EXPECT_THROW(network.save(missing_path / "model.json"), std::runtime_error);
}
