#include <cmath>
#include <gtest/gtest.h>

#include <boost/math/tools/promotion.hpp>

#include "stan/agrad/fvar.hpp"
#include "stan/math/functor/apply.hpp"
#include "stan/agrad/fwd/functor/apply.hpp"

// f(x) = exp(x)
// d/dx f(x) = exp(x)
class exp_fun {
public:
  static inline double f(double x) {
    return std::exp(x);
  }
  template <typename T>
  static inline
  T dx(const T& /*x*/, const T& fx) {
    return fx;
  }
};


TEST(MathFunctorApply,val_fvar_dbl) {
  using stan::agrad::fvar;
  using stan::math::apply; 
  double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(apply<exp_fun>(fvar<double>(nan,1.7)).val_));
  EXPECT_FLOAT_EQ(std::exp(-3), apply<exp_fun>(fvar<double>(-3,1.7)).val_);
}
TEST(MathFunctorApply,dx_fvar_dbl) {
  using stan::agrad::fvar;
  using stan::math::apply; 
  EXPECT_FLOAT_EQ(1.7 * std::exp(-3), 
                  apply<exp_fun>(fvar<double>(-3,1.7)).d_);
}

TEST(MathFunctorApply,val_fvar_fvar_dbl) {
  using stan::agrad::fvar;
  using stan::math::apply; 
  fvar<fvar<double> > x(-3,1.7);
  fvar<fvar<double> > fx = apply<exp_fun>(x);
  EXPECT_FLOAT_EQ(std::exp(-3), fx.val_.val_);
}

TEST(MathFunctorApply,dx_ffd_fvar_fvar) {
  using stan::agrad::fvar;
  using stan::math::apply; 
  fvar<fvar<double> > x(fvar<double>(-2,1.0), 
                        fvar<double>(0,0));
  fvar<fvar<double> > y(fvar<double>(1.9,0.0),
                        fvar<double>(1.0,0));
  fvar<fvar<double> > exp_xy = apply<exp_fun>(x * y);
  EXPECT_FLOAT_EQ(exp(-2 * 1.9), exp_xy.val_.val_);      // exp(xy)
  EXPECT_FLOAT_EQ(exp(-2 * 1.9) * 1.9, exp_xy.val_.d_);  // d/dx exp(xy)
  EXPECT_FLOAT_EQ(exp(-2 * 1.9) * -2, exp_xy.d_.val_);   // d/dy exp(xy)
  EXPECT_FLOAT_EQ(exp(-2 * 1.9) * -2 * 1.9
                  + exp(-2 * 1.9), 
                  exp_xy.d_.d_);                         // d^2/dx.dy exp(xy)
}       
TEST(MathFunctorApply,dx_ffd_fvar_dbl) {
  using stan::agrad::fvar;
  using stan::math::apply; 
  fvar<fvar<double> > x(fvar<double>(-2,1.0), 
                        fvar<double>(0,0));
  double y = 1.9;
  fvar<fvar<double> > exp_xy = apply<exp_fun>(x * y);
  EXPECT_FLOAT_EQ(exp(-2 * 1.9), exp_xy.val_.val_);      // exp(xy)
  EXPECT_FLOAT_EQ(exp(-2 * 1.9) * 1.9, exp_xy.val_.d_);  // d/dx exp(xy)
  EXPECT_FLOAT_EQ(0, exp_xy.d_.val_);                    // d/dy exp(xy)
  EXPECT_FLOAT_EQ(0,exp_xy.d_.d_);                       // d^2/dx.dy exp(xy)
}       
TEST(MathFunctorApply,dx_ffd_dbl_fvar) {
  using stan::agrad::fvar;
  using stan::math::apply; 
  double x = -2;
  fvar<fvar<double> > y(fvar<double>(1.9,0.0),
                        fvar<double>(1.0,0));
  fvar<fvar<double> > exp_xy = apply<exp_fun>(x * y);
  EXPECT_FLOAT_EQ(exp(-2 * 1.9), exp_xy.val_.val_);      // exp(xy)
  EXPECT_FLOAT_EQ(0, exp_xy.val_.d_);                    // d/dx exp(xy)
  EXPECT_FLOAT_EQ(exp(-2 * 1.9) * -2, exp_xy.d_.val_);   // d/dy exp(xy)
  EXPECT_FLOAT_EQ(0, exp_xy.d_.d_);                      // d^2/dx.dy exp(xy)
}       


// f(x1,x2) = sqrt(x1^2 + x2^2)
// d/dx1 f(x1,x2) = x1 / f(x1,x2)
// d/dx2 f(x1,x2) = x2 / f(x1,x2)
struct hypot_fun {
  static inline double f(double x1, double x2) {
    return std::sqrt(x1 * x1 + x2 * x2);
  }
  template <typename T1, typename T2, typename T>
  static inline T dx1(const T1& x1, const T2& /*x2*/, const T& fx) {
    return x1 / fx;
  }
  template <typename T1, typename T2, typename T>
  static inline T dx2(const T1& /*x1*/, const T2& x2, const T& fx) {
    return x2 / fx;
  }
};


// hypot(fvar<double>,fvar<double>)
TEST(MathFunctorApply,hypot_fd_fd) {
  typedef stan::agrad::fvar<double> fd;
  using stan::math::apply; 
  fd x1(3.0,1);
  fd x2(4.0,0);
  fd fx = apply<hypot_fun>(x1,x2);
  EXPECT_FLOAT_EQ(5.0, fx.val_);
  EXPECT_FLOAT_EQ(3.0 / 5.0, fx.d_);

  x1 = fd(3.0,0);
  x2 = fd(4.0,1);
  fx = apply<hypot_fun>(x1,x2);
  EXPECT_FLOAT_EQ(5.0, fx.val_);
  EXPECT_FLOAT_EQ(4.0 / 5.0, fx.d_);
}
// hypot(fvar<double>,double)
TEST(MathFunctorApply,hypot_fd_d) {
  typedef stan::agrad::fvar<double> fd;
  using stan::math::apply; 
  fd x1(3.0,1);
  double x2 = 4;
  fd fx = apply<hypot_fun>(x1,x2);
  EXPECT_FLOAT_EQ(5.0, fx.val_);
  EXPECT_FLOAT_EQ(3.0 / 5.0, fx.d_);
}
// hypot(double,fvar<double>)
TEST(MathFunctorApply,hypot_d_fd) {
  typedef stan::agrad::fvar<double> fd;
  using stan::math::apply; 
  double x1 = 3;
  fd x2(4.0,1);
  fd fx = apply<hypot_fun>(x1,x2);
  EXPECT_FLOAT_EQ(5.0, fx.val_);
  EXPECT_FLOAT_EQ(4.0 / 5.0, fx.d_);
}


TEST(MathFunctorApply,hypot_ffd_ffd) {
  typedef stan::agrad::fvar<double> fd;
  typedef stan::agrad::fvar<fd> ffd;
  using stan::math::apply; 
  ffd x1(fd(3.0,1),fd(0,0));  // d/dx1 = 1
  ffd x2(fd(4.0,0),fd(1,0));  // d/dx2 = 1
  ffd fx = apply<hypot_fun>(x1,x2);
  EXPECT_FLOAT_EQ(5.0, fx.val_.val_);   // fx
  EXPECT_FLOAT_EQ(3.0 / 5.0, fx.val_.d_);  // d/dx1 fx
  EXPECT_FLOAT_EQ(4.0 / 5.0, fx.d_.val_);  // d/dx2 fx
  EXPECT_FLOAT_EQ(-3.0 * 4.0 * std::pow(3.0 * 3.0 + 4.0 * 4.0,-1.5), fx.d_.d_);
}
TEST(MathFunctorApply,hypot_ffd_d) {
  typedef stan::agrad::fvar<double> fd;
  typedef stan::agrad::fvar<fd> ffd;
  using stan::math::apply; 
  ffd x1(fd(3.0,1),fd(0,0));  // d/dx1 = 1
  double x2 = 4;              // d/dx2 = 0
  ffd fx = apply<hypot_fun>(x1,x2);
  EXPECT_FLOAT_EQ(5.0, fx.val_.val_);   // fx
  EXPECT_FLOAT_EQ(3.0 / 5.0, fx.val_.d_);  // d/dx1 fx
  EXPECT_FLOAT_EQ(0, fx.d_.val_);  // d/dx2 fx
  EXPECT_FLOAT_EQ(0, fx.d_.d_);
}
TEST(MathFunctorApply,hypot_d_ffd) {
  typedef stan::agrad::fvar<double> fd;
  typedef stan::agrad::fvar<fd> ffd;
  using stan::math::apply; 
  double x1 = 3;              // d/dx1 = 0
  ffd x2(fd(4.0,0),fd(1,0));  // d/dx2 = 1
  ffd fx = apply<hypot_fun>(x1,x2);
  EXPECT_FLOAT_EQ(5.0, fx.val_.val_);   // fx
  EXPECT_FLOAT_EQ(0, fx.val_.d_);  // d/dx1 fx
  EXPECT_FLOAT_EQ(4.0 / 5.0, fx.d_.val_);  // d/dx2 fx
  EXPECT_FLOAT_EQ(0, fx.d_.d_);
}

