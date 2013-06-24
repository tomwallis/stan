#include <vector>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <stan/agrad/agrad.hpp>
#include <stan/prob/transform/identity_constrain.hpp>
#include <stan/prob/transform/identity_free.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform,identity) {
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_constrain(4.0));
}
TEST(prob_transform,identity_j) {
  double lp = 1.0;
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_constrain(4.0,lp));
  EXPECT_FLOAT_EQ(1.0,lp);
}
TEST(prob_transform,identity_free) {
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_free(4.0));
}
TEST(prob_transform,identity_rt) {
  double x = 1.2;
  double xc = stan::prob::identity_constrain(x);
  double xcf = stan::prob::identity_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);

  double y = -1.0;
  double yf = stan::prob::identity_free(y);
  double yfc = stan::prob::identity_constrain(yf);
  EXPECT_FLOAT_EQ(y,yfc);
}
