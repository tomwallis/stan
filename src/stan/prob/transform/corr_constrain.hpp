#ifndef __STAN__PROB__TRANSFORM_CORR_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_CORR_CONSTRAIN_HPP__

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <boost/multi_array.hpp>
#include <boost/throw_exception.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/agrad/matrix.hpp>
#include <stan/math.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/matrix/validate_less.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix_error_handling.hpp>

#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

namespace stan {
  
  namespace prob {

    /**
     * Return the result of transforming the specified scalar to have
     * a valid correlation value between -1 and 1 (inclusive).
     *
     * <p>The transform used is the hyperbolic tangent function,
     *
     * <p>\f$f(x) = \tanh x = \frac{\exp(2x) - 1}{\exp(2x) + 1}\f$.
     * 
     * @param x Scalar input.
     * @return Result of transforming the input to fall between -1 and 1.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_constrain(const T x) {
      return tanh(x);
    }

    /**
     * Return the result of transforming the specified scalar to have
     * a valid correlation value between -1 and 1 (inclusive).
     *
     * <p>The transform used is as specified for
     * <code>corr_constrain(T)</code>.  The log absolute Jacobian
     * determinant is
     *
     * <p>\f$\log | \frac{d}{dx} \tanh x  | = \log (1 - \tanh^2 x)\f$.
     * 
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_constrain(const T x, T& lp) {
      using stan::math::log1m;
      T tanh_x = tanh(x);
      lp += log1m(tanh_x * tanh_x);
      return tanh_x;
    }

  }

}
#endif
