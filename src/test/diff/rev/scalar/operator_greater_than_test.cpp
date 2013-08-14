#include <stan/diff/rev/scalar/operator_greater_than.hpp>
#include <test/diff/util.hpp>
#include <gtest/gtest.h>

TEST(DiffRevScalar,a_gt_b) {
  AVAR a = 5.0;
  AVAR b = 6.0;
  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);
  AVAR c = 6.0;
  EXPECT_FALSE(b > c);
  EXPECT_FALSE(c > b);
}

TEST(DiffRevScalar,a_gt_y) {
  AVAR a = 6.0;
  double y = 5.0;
  EXPECT_TRUE(a > y);
  EXPECT_FALSE(y > a);
  AVAR c = 6.0;
  EXPECT_FALSE(a > c);
  EXPECT_FALSE(c > a);
}

TEST(DiffRevScalar,x_gt_b) {
  double x = 6.0;
  AVAR b = 5.0;
  EXPECT_TRUE(x > b);
  EXPECT_FALSE(b > x);
  double y = 5.0;
  EXPECT_FALSE(b > y);
  EXPECT_FALSE(y > b);
}
