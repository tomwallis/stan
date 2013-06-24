#include <vector>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include <stan/agrad/agrad.hpp>
#include <stan/prob/transform/cov_matrix_constrain.hpp>
#include <stan/prob/transform/cov_matrix_free.hpp>
#include <stan/math/matrix/determinant.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform,cov_matrix_constrain_exception) {
  Matrix<double,Dynamic,1> x(7);
  int K = 12;
  EXPECT_THROW(stan::prob::cov_matrix_constrain(x,K), std::domain_error);
}
TEST(prob_transform,cov_matrix_free_exception) {
  Matrix<double,Dynamic,Dynamic> y(0,0);
  
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(0,10);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(10,0);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(1,2);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);

  y.resize(2,2);
  y << 0, 0, 0, 0;
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
}

TEST(prob_transform,cov_matrix_rt) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2 + K);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5,
    1.0, 2.0, -1.5, 2.5;
  Matrix<double,Dynamic,Dynamic> y = stan::prob::cov_matrix_constrain(x,K);
  Matrix<double,Dynamic,1> xrt = stan::prob::cov_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (Matrix<double,Dynamic,1>::size_type i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,cov_matrix_jacobian) {
  using stan::agrad::var;
  using stan::math::determinant;
  using std::log;
  using std::fabs;

  Matrix<var,Dynamic,Dynamic>::size_type K = 4;
  //unsigned int K = 4;
  unsigned int K_choose_2 = 6;
  Matrix<var,Dynamic,1> X(K_choose_2 + K);
  X << 1.0, 2.0, -3.0, 1.7, 9.8, 
    -12.2, 0.4, 0.2, 1.2, 2.7;
  std::vector<var> x;
  for (int i = 0; i < X.size(); ++i)
    x.push_back(X(i));
  var lp = 0.0;
  Matrix<var,Dynamic,Dynamic> Sigma = stan::prob::cov_matrix_constrain(X,K,lp);
  std::vector<var> y;
  for (Matrix<var,Dynamic,Dynamic>::size_type m = 0; m < K; ++m)
    for (Matrix<var,Dynamic,Dynamic>::size_type n = 0; n <= m; ++n)
      y.push_back(Sigma(m,n));

  std::vector<std::vector<double> > j;
  stan::agrad::jacobian(y,x,j);

  Matrix<double,Dynamic,Dynamic> J(10,10);
  for (int m = 0; m < 10; ++m)
    for (int n = 0; n < 10; ++n)
      J(m,n) = j[m][n];

  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det,lp.val());
}
