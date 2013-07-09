#ifndef __STAN__AGRAD__FWD__FUNCTOR__APPLY_HPP__
#define __STAN__AGRAD__FWD__FUNCTOR__APPLY_HPP__

#include <stan/agrad/fwd/fvar.hpp>

namespace stan {
  namespace agrad {

    /**
     * Apply functor specified by template class to specified
     * argument.
     *
     * The class <code>F</code> specified by the template parameter
     * applying to scalar type <code>T</code> must implement a static
     * function with signatures:
     *
     * Function value:
     * <code>double f(double x);</code>
     *
     * Derivative at x:
     * <code>T dx(const T& x, const T& fx);</code>
     *
     * Note that the arguments are <code>double</code> despite the
     * generic template type.  If <code>T</code> is
     * <code>double</code>, the signature <code>double
     * dx(double,double)</code> may be implemented instead of the above.
     *
     * The arguments to <code>dx</code> are the values of x and of of
     * f(x), the latter of which may be used to help calculate the
     * derivative.
     *
     * @tparam F Class defining static function f(double)
     * @tparam T Scalar type for forward-mode algorithmic differentiation
     * @param x Argument
     * @return Result of applying static function to argument,
     * F::f(x), with all derivatives calculated
     */
    template <class F, typename T>
    inline fvar<T> apply(const fvar<T>& x) {
      using stan::math::apply; 
      T fx = apply<F>(x.val_);  
      return fvar<T>(fx, x.d_ * F::dx(x.val_, fx));
    }

    template<class F, typename T>
    inline fvar<T> apply(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::apply;
      T fx = apply<F>(x1.val_,x2.val_);
      return fvar<T>(fx, 
                     x1.d_ * F::dx1(x1.val_,x2.val_,fx) 
                     + x2.d_ * F::dx2(x1.val_,x2.val_,fx));
    }
    template<class F, typename T>
    inline fvar<T> apply(const fvar<T>& x1, double x2) {
      using stan::math::apply;
      T fx = apply<F>(x1.val_,x2);
      return fvar<T>(fx, x1.d_ * F::dx1(x1.val_,x2,fx));
    }
    template<class F, typename T>
    inline fvar<T> apply(double x1, const fvar<T>& x2) {
      using stan::math::apply;
      T fx = apply<F>(x1,x2.val_);
      return fvar<T>(fx, x2.d_ * F::dx2(x1,x2.val_,fx));
    }

  }
}

#endif
