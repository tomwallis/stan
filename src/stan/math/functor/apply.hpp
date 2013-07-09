#ifndef __STAN__MATH__FUNCTOR__APPLY_HPP__
#define __STAN__MATH__FUNCTOR__APPLY_HPP__

namespace stan {
  namespace math {

    /**
     * Apply functor specified by template class to specified
     * argument.  
     *
     * The class <code>F</code> specified by the template parameter
     * must implement a static function with signature:
     *
     * <code>double f(double x);</code>
     *
     * @tparam F Class defining static function f(double)
     * @param x Argument
     * @return Result of applying static function to argument, F::f(x)
     */
    template <class F>
    inline double apply(double x) {
      return F::f(x);
    }

    /**
     * Apply functor specified by template class to specified
     * argument.  
     *
     * The class <code>F</code> specified by the template parameter
     * must implement a static function with signature:
     *
     * <code>double f(double x1, double x2);</code>
     *
     * @tparam F Class defining function
     * @param x1 First argument
     * @param x2 Second argument
     * @return Result of applying static function to argument, F::f(x1,x2)
     */
    template <class F>
    inline double apply(double x1, double x2) {
      return F::f(x1,x2);
    }

    // template <class F>
    // inline std::vector<double> apply(const std::vector<double>& x) {
    //   std::vector<double> y;
    //   ys.reserve(x.size());
    //   for (size_t i = 0; i < x.size(); ++i)
    //     y[i] = apply<F>(x[i]);
    //   return y;
    // }

    // template <class F, int R, int C>
    // inline Eigen::Matrix<double,R,C> apply(const Eigen::Matrix<double,R,C>& x) {
    //   Eigen::Matrix<double,R,C> y(x.rows(),x.cols());
    //   for (int i = 0; i < x.size(); ++i)
    //     y(i) = apply<F>(x(i));
    //   return y;
    // }

  }
}

#endif
