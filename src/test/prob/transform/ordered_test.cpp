#include <vector>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <stan/agrad/agrad.hpp>
#include <stan/prob/transform/ordered_constrain.hpp>
#include <stan/prob/transform/ordered_free.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform,ordered) {
  Matrix<double,Dynamic,1> x(3);
  x << -15.0, -2.0, -5.0;
  Matrix<double,Dynamic,1> y = stan::prob::ordered_constrain(x);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(-15.0, y[0]);
  EXPECT_EQ(-15.0 + exp(-2.0), y[1]);
  EXPECT_EQ(-15.0 + exp(-2.0) + exp(-5.0), y[2]);
}
TEST(prob_transform,ordered_j) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -2.0, -5.0;
  double lp = -152.1;
  Matrix<double,Dynamic,1> y = stan::prob::ordered_constrain(x,lp);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(1.0, y[0]);
  EXPECT_EQ(1.0 + exp(-2.0), y[1]);
  EXPECT_EQ(1.0 + exp(-2.0) + exp(-5.0), y[2]);
  EXPECT_EQ(-152.1 - 2.0 - 5.0,lp);
}
TEST(prob_transform,ordered_f) {
  Matrix<double,Dynamic,1> y(3);
  y << -12.0, 1.1, 172.1;
  Matrix<double,Dynamic,1> x = stan::prob::ordered_free(y);
  EXPECT_EQ(y.size(),x.size());
  EXPECT_FLOAT_EQ(-12.0, x[0]);
  EXPECT_FLOAT_EQ(log(1.1 + 12.0), x[1]);
  EXPECT_FLOAT_EQ(log(172.1 - 1.1), x[2]);
}
TEST(prob_transform,ordered_f_exception) {
  Matrix<double,Dynamic,1> y(3);
  y << -0.1, 0.0, 1.0;
  EXPECT_NO_THROW(stan::prob::ordered_free(y));
  y << 0.0, 0.0, 0.0;
  EXPECT_THROW(stan::prob::ordered_free(y), std::domain_error);
  y << 0.0, 1, 0.9;
  EXPECT_THROW(stan::prob::ordered_free(y), std::domain_error);
}
TEST(prob_transform,ordered_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << -1.0, 8.0, -3.9;
  Matrix<double,Dynamic,1> y = stan::prob::ordered_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::prob::ordered_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (Matrix<double,Dynamic,1>::size_type i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,ordered_jacobian_ad) {
  using stan::agrad::var;
  using stan::prob::ordered_constrain;
  using stan::math::determinant;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double,Dynamic,1> x(3);
  x << -12.0, 3.0, -1.9;
  double lp = 0.0;
  Matrix<double,Dynamic,1> y = ordered_constrain(x,lp);

  Matrix<var,Dynamic,1> xv(3);
  xv << -12.0, 3.0, -1.9;

  std::vector<var> xvec(3);
  for (int i = 0; i < 3; ++i)
    xvec[i] = xv[i];

  Matrix<var,Dynamic,1> yv = ordered_constrain(xv);


  EXPECT_EQ(y.size(), yv.size());
  for (int i = 0; i < y.size(); ++i)
    EXPECT_FLOAT_EQ(y(i),yv(i).val());

  std::vector<var> yvec(3);
  for (unsigned int i = 0; i < 3; ++i)
    yvec[i] = yv[i];

  std::vector<std::vector<double> > j;
  stan::agrad::jacobian(yvec,xvec,j);

  Matrix<double,Dynamic,Dynamic> J(3,3);
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n)
      J(m,n) = j[m][n];
  
  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det, lp);
}
