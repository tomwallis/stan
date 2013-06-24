#include <vector>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <stan/agrad/agrad.hpp>
#include <stan/prob/transform/prob_constrain.hpp>
#include <stan/prob/transform/prob_free.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform, prob) {
  EXPECT_FLOAT_EQ(stan::math::inv_logit(-1.0), 
                  stan::prob::prob_constrain(-1.0));
}
TEST(prob_transform, prob_j) {
  double lp = -17.0;
  double L = 0.0;
  double U = 1.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(L + (U - L) * stan::math::inv_logit(x), 
                  stan::prob::prob_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(stan::math::inv_logit(x)) 
                  + log(1.0 - stan::math::inv_logit(x)),
                  lp);
}
TEST(prob_transform, prob_f) {
  double L = 0.0;
  double U = 1.0;
  double y = 0.4;
  EXPECT_FLOAT_EQ(stan::math::logit((y - L) / (U - L)),
                  stan::prob::prob_free(y));
}
TEST(prob_transform, prob_f_exception) {
  EXPECT_THROW (stan::prob::prob_free(1.1), std::domain_error);
  EXPECT_THROW (stan::prob::prob_free(-0.1), std::domain_error);
}
TEST(prob_transform, prob_rt) {
  double x = -1.0;
  double xc = stan::prob::prob_constrain(x);
  double xcf = stan::prob::prob_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::prob_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}
