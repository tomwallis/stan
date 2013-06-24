#include <vector>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <stan/agrad/agrad.hpp>
#include <stan/prob/transform/positive_constrain.hpp>
#include <stan/prob/transform/positive_free.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform, positive) {
  EXPECT_FLOAT_EQ(exp(-1.0), stan::prob::positive_constrain(-1.0));
}
TEST(prob_transform, positive_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(exp(-1.0), stan::prob::positive_constrain(-1.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);
}
TEST(prob_transform, positive_f) {
  EXPECT_FLOAT_EQ(log(0.5), stan::prob::positive_free(0.5));
}
TEST(prob_transform, positive_f_exception) {
  EXPECT_THROW (stan::prob::positive_free(-1.0), std::domain_error);
}
TEST(prob_transform, positive_rt) {
  double x = -1.0;
  double xc = stan::prob::positive_constrain(x);
  double xcf = stan::prob::positive_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::positive_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}
