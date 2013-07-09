#include <cmath>

#include "stan/math/functor/apply.hpp"

#include <gtest/gtest.h>

class exp_fun {
public:
  static inline 
  double f(const double x) {
    return std::exp(x);
  }
};

struct hypot_fun {
  static inline double f(double x1, double x2) {
    return std::sqrt(x1 * x1 + x2 * x2);
  }
  static inline double dx1(double /*x1*/, double x2, double fx) {
    return x2 / fx;
  }
  static inline double dx2(double x1, double /*x2*/, double fx) {
    return x1 / fx;
  }
};


TEST(MathFunctorApply,unary) {
  using stan::math::apply;
  EXPECT_TRUE(std::isnan(apply<exp_fun>(std::numeric_limits<double>::quiet_NaN())));
  EXPECT_FLOAT_EQ(std::exp(-3), apply<exp_fun>(-3));
  EXPECT_FLOAT_EQ(std::exp(0), apply<exp_fun>(0));
  EXPECT_FLOAT_EQ(std::exp(1), apply<exp_fun>(1));
}

TEST(MathFunctorApply,binary) {
  using stan::math::apply;
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(1.2, std::numeric_limits<double>::quiet_NaN())));
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(std::numeric_limits<double>::quiet_NaN(),1.2)));
  EXPECT_FLOAT_EQ(std::sqrt(-3 * -3 + 1.9 * 1.9),
                  apply<hypot_fun>(-3,1.9));
}
