#include <gtest/gtest.h>

#include <nnl/activation.h>

TEST(ActivationTest, ComputesExpectedValues) {
    EXPECT_DOUBLE_EQ(act_sigmoid(0.0), 0.5);
    EXPECT_DOUBLE_EQ(act_sigmoid_prim(0.5), 0.25);
    EXPECT_DOUBLE_EQ(act_sigmoid_bi(0.0), 0.0);
    EXPECT_DOUBLE_EQ(act_sigmoid_bi_prim(0.0), 0.5);
    EXPECT_DOUBLE_EQ(act_linear(-2.5), -2.5);
    EXPECT_DOUBLE_EQ(act_linear_prim(3.0), 1.0);
    EXPECT_DOUBLE_EQ(act_jump(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(act_jump(0.0), 1.0);
    EXPECT_DOUBLE_EQ(act_jump(1.0), 1.0);
    EXPECT_DOUBLE_EQ(act_jump_prim(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(act_jump_prim(0.0), 0.0);
    EXPECT_DOUBLE_EQ(act_jump_prim(1.0), 0.0);
}
