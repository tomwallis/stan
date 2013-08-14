#include <stan/math/scalar/log2.hpp>
#include <gtest/gtest.h>

TEST(MathScalar, log2_fun) {
  using stan::math::log2;

  EXPECT_FLOAT_EQ(std::log(2.0), log2());
}

TEST(MathScalarLog2, log2) {
  using stan::math::log2;

  EXPECT_FLOAT_EQ(1.0, log2(2.0));
  EXPECT_FLOAT_EQ(2.0, log2(4.0));
  EXPECT_FLOAT_EQ(3.0, log2(8.0));
}
