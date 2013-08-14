#include <stan/diff/rev/scalar/is_uninitialized.hpp>
#include <gtest/gtest.h>

TEST(DiffRevScalar,undefined) {
  stan::diff::var a;
  EXPECT_TRUE(a.is_uninitialized());
  a = 5;
  EXPECT_FALSE(a.is_uninitialized());
}
