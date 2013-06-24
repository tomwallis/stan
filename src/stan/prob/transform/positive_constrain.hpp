#ifndef __STAN__PROB__TRANSFORM_POSITIVE_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_POSITIVE_CONSTRAIN_HPP__

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
     * Return the positive value for the specified unconstrained input.
     *
     * <p>The transform applied is
     *
     * <p>\f$f(x) = \exp(x)\f$.
     * 
     * @param x Arbitrary input scalar.
     * @return Input transformed to be positive.
     */
    template <typename T> 
    inline
    T positive_constrain(const T x) {
      return exp(x);
    }

    /**
     * Return the positive value for the specified unconstrained input,
     * incrementing the scalar reference with the log absolute
     * Jacobian determinant.
     *
     * <p>See <code>positive_constrain(T)</code> for details
     * of the transform.  The log absolute Jacobian determinant is
     *
     * <p>\f$\log | \frac{d}{dx} \mbox{exp}(x) | 
     *    = \log | \mbox{exp}(x) | =  x\f$.
     * 
     * @param x Arbitrary input scalar.
     * @param lp Log probability reference.
     * @return Input transformed to be positive.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T positive_constrain(const T x, T& lp) {
      lp += x;
      return exp(x); 
    }

  }

}
#endif
