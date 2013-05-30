#include <stan/agrad/fwd/value_of.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,value_of) {
  using stan::agrad::fvar;
  using stan::math::value_of;
  using stan::agrad::value_of;

  fvar<double> a(5.0);
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  EXPECT_FLOAT_EQ(5.0, value_of(5.0)); // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/value_of.hpp>

TEST(AgradRev,value_of_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::value_of;
  using stan::agrad::value_of;
  
  fvar<var> a(5.0);
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  var b(5.0);
  EXPECT_FLOAT_EQ(5.0, value_of(b)); // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}
