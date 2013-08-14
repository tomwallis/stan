#include <stan/math/scalar/log_sum_exp.hpp>
#include <gtest/gtest.h>

void test_log_sum_exp(double a, double b) {
  using std::log;
  using std::exp;
  using stan::math::log_sum_exp;
  EXPECT_FLOAT_EQ(log(exp(a) + exp(b)),
                  log_sum_exp(a,b));
}

TEST(MathScalar, log_sum_exp_2) {
  using stan::math::log_sum_exp;
  test_log_sum_exp(1.0,2.0);
  test_log_sum_exp(1.0,1.0);
  test_log_sum_exp(3.0,2.0);
  test_log_sum_exp(-20.0,12);
  test_log_sum_exp(-20.0,12);

  // exp(10000.0) overflows
  EXPECT_FLOAT_EQ(10000.0,log_sum_exp(10000.0,0.0));
  EXPECT_FLOAT_EQ(0.0,log_sum_exp(-10000.0,0.0));
}
