#include <stan/diff/rev/scalar/log_sum_exp.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>
#include <stan/diff.hpp>

TEST(DiffRevScalar,log_sum_exp_vv) {
  AVAR a = 5.0;
  AVAR b = 2.0;
  AVAR f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) + std::exp(2)), f.val());
  
  AVEC x = createAVEC(a, b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[1]);

  // underflow example
  a = 1000;
  b = 10;
  f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(a, b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[0]);
  EXPECT_FLOAT_EQ (std::exp (10.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[1]);
}
TEST(DiffRevScalar,log_sum_exp_vd) {
  AVAR a = 5.0;
  double b = 2.0;
  AVAR f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) + std::exp(2)), f.val());

  AVEC x = createAVEC(a);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(5.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);

  // underflow example
  a = 1000;
  b = 10;
  f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(a);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[0]);
}
TEST(DiffRevScalar,log_sum_exp_dv) {
  double a = 5.0;
  AVAR b = 2.0;
  AVAR f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(5) + std::exp(2)), f.val());
  
  AVEC x = createAVEC(b);
  VEC grad_f;
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ(std::exp(2.0) / (std::exp(5.0) + std::exp(2.0)), grad_f[0]);

  // underflow example
  a = 10;
  b = 1000;
  f = log_sum_exp(a, b);
  EXPECT_FLOAT_EQ (std::log(std::exp(0.0) + std::exp(-990.0)) + 1000.0, f.val());
  
  x = createAVEC(b);
  f.grad(x,grad_f);
  EXPECT_FLOAT_EQ (std::exp (1000.0 - (std::log(std::exp(0.0) + std::exp(-999.0)) + 1000)), grad_f[0]);
}
