#include <cmath>

#include <boost/math/tools/promotion.hpp>

#include "stan/math/functor/apply.hpp"
#include "stan/agrad/rev/var.hpp"
#include "stan/agrad/rev/functor/apply.hpp"
#include "stan/agrad/rev/operator_multiplication.hpp"

#include <gtest/gtest.h>

class exp_fun {
public:
  static inline 
  double f(double x) {
    return std::exp(x);
  }
  static inline
  double dx(double /*x*/, double fx) {
    return fx;
  }
};
template <typename T>
inline typename boost::math::tools::promote_args<T>::type
my_exp(const T& x) {
  using stan::math::apply;
  return apply<exp_fun>(x);
}

struct hypot_fun {
  static inline double f(double x1, double x2) {
    return std::sqrt(x1 * x1 + x2 * x2);
  }
  static inline double dx1(double x1, double /*x2*/, double fx) {
    return x1 / fx;
  }
  static inline double dx2(double /*x1*/, double x2, double fx) {
    return x2 / fx;
  }
};
template <typename T1, typename T2>
inline typename boost::math::tools::promote_args<T1,T2>::type
my_hypot(const T1& x1, const T2&x2) {
  using stan::math::apply;
  return apply<hypot_fun>(x1,x2);
}

TEST(MathFunctorApply,val_unary) {
  using stan::agrad::var;
  using stan::math::apply;  // use arg-dependent lookup to find stan::agrad::apply
  EXPECT_TRUE(std::isnan(my_exp(var(std::numeric_limits<double>::quiet_NaN())).val()));
  EXPECT_FLOAT_EQ(1.7 * std::exp(-3), (1.7 * my_exp(var(-3))).val());
  EXPECT_FLOAT_EQ(1.7 * std::exp(0), (1.7 * my_exp(var(0))).val());
  EXPECT_FLOAT_EQ(1.7 * std::exp(1), (1.7 * my_exp(var(1))).val());
}
TEST(MathFunctorApply,grad_unary) {
  using std::vector;
  using stan::agrad::var;
  using stan::math::apply;
  
  std::vector<var> xs;
  var x = -3;
  xs.push_back(x);
  var fx = 1.7 * my_exp(x);

  std::vector<double> g;
  fx.grad(xs,g);
  EXPECT_EQ(1,g.size());
  EXPECT_FLOAT_EQ(1.7 * std::exp(-3), g[0]);
}
TEST(MathFunctorApply,val_binary_vv) {
  using stan::agrad::var;
  using stan::math::apply;

  double nan_dbl = std::numeric_limits<double>::quiet_NaN();
  var nan_var = var(nan_dbl);

  EXPECT_TRUE(std::isnan(apply<hypot_fun>(nan_var,nan_var).val()));
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(nan_var,var(1.2)).val()));
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(var(1.2),nan_var).val()));
  EXPECT_FLOAT_EQ(1.7 * 5, (1.7 * apply<hypot_fun>(var(3),var(4))).val());
}
TEST(MathFunctorApply,val_binary_vd) {
  using stan::agrad::var;
  using stan::math::apply;

  double nan_dbl = std::numeric_limits<double>::quiet_NaN();
  var nan_var = var(nan_dbl);

  EXPECT_TRUE(std::isnan(apply<hypot_fun>(nan_var,nan_dbl).val()));
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(nan_var,1.0).val()));
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(var(1.2),nan_dbl).val()));
  EXPECT_FLOAT_EQ(1.7 * 5, (1.7 * apply<hypot_fun>(var(3),4.0)).val());
}
TEST(MathFunctorApply,val_binary_dv) {
  using stan::agrad::var;
  using stan::math::apply;

  double nan_dbl = std::numeric_limits<double>::quiet_NaN();
  var nan_var = var(nan_dbl);

  EXPECT_TRUE(std::isnan(apply<hypot_fun>(nan_dbl,nan_var).val()));
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(nan_var,var(1.2)).val()));
  EXPECT_TRUE(std::isnan(apply<hypot_fun>(var(1.2),nan_var).val()));
  EXPECT_FLOAT_EQ(1.7 * 5, (1.7 * apply<hypot_fun>(3.0,var(4))).val());
}

TEST(MathFunctorApply,grad_binary_vv) {
  using stan::agrad::var;
  using stan::math::apply;
  using std::vector;
  using std::sqrt;

  var x1 = 3;
  var x2 = 4;
  vector<var> x;
  x.push_back(x1);
  x.push_back(x2);

  var fx = 14.23 * apply<hypot_fun>(x1,x2);
  vector<double> g;
  fx.grad(x,g);
  EXPECT_FLOAT_EQ(14.23 * 3 * 2 * 0.5 / sqrt(3 * 3 + 4 * 4),
                  g[0]);
  EXPECT_FLOAT_EQ(14.23 * 4 * 2 * 0.5 / sqrt(3 * 3 + 4 * 4),
                  g[1]);
}
TEST(MathFunctorApply,grad_binary_vd) {
  using stan::agrad::var;
  using stan::math::apply;
  using std::vector;
  using std::sqrt;

  var x1 = 3;
  double x2 = 4;
  vector<var> x;
  x.push_back(x1);

  var fx = 14.23 * apply<hypot_fun>(x1,x2);
  vector<double> g;
  fx.grad(x,g);
  EXPECT_FLOAT_EQ(14.23 * 3 * 2 * 0.5 / sqrt(3 * 3 + 4 * 4),
                  g[0]);
}
TEST(MathFunctorApply,grad_binary_dv) {
  using stan::agrad::var;
  using stan::math::apply;
  using std::vector;
  using std::sqrt;

  double x1 = 3;
  var x2 = 4;
  vector<var> x;
  x.push_back(x2);

  var fx = 14.23 * apply<hypot_fun>(x1,x2);
  vector<double> g;
  fx.grad(x,g);
  EXPECT_FLOAT_EQ(14.23 * 4 * 2 * 0.5 / sqrt(3 * 3 + 4 * 4),
                  g[0]);
}
