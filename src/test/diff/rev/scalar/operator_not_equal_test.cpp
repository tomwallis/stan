#include <stan/diff/rev/scalar/operator_not_equal.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRevScalar,a_neq_y) {
  AVAR a = 2.0;
  double y = 3.0;
  EXPECT_TRUE(a != y);
  EXPECT_TRUE(y != a);
  double z = 2.0;
  EXPECT_FALSE(a != z);
  EXPECT_FALSE(z != a);
}
