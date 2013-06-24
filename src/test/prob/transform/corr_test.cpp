#include <vector>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <stan/agrad/agrad.hpp>
#include <stan/prob/transform/corr_constrain.hpp>
#include <stan/prob/transform/corr_free.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform, corr) {
  EXPECT_FLOAT_EQ(std::tanh(-1.0), 
                  stan::prob::corr_constrain(-1.0));
}
TEST(prob_transform, corr_j) {
  double lp = -17.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(std::tanh(x), 
                  stan::prob::corr_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + (log(1.0 - std::tanh(x) * std::tanh(x))),
                  lp);
}
TEST(prob_transform, corr_f) {
  EXPECT_FLOAT_EQ(atanh(-0.4), 0.5 * std::log((1.0 + -0.4)/(1.0 - -0.4)));
  double y = -0.4;
  EXPECT_FLOAT_EQ(atanh(y),
                  stan::prob::corr_free(y));
}
TEST(prob_transform, corr_rt) {
  double x = -1.0;
  double xc = stan::prob::corr_constrain(x);
  double xcf = stan::prob::corr_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::corr_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}
