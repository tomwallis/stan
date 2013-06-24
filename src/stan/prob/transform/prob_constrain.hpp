#ifndef __STAN__PROB__TRANSFORM_PROB_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_PROB_CONSTRAIN_HPP__

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
     * Return a probability value constrained to fall between 0 and 1
     * (inclusive) for the specified free scalar.
     *
     * <p>The transform is the inverse logit,
     *
     * <p>\f$f(x) = \mbox{logit}^{-1}(x) = \frac{1}{1 + \exp(x)}\f$.
     *
     * @param x Free scalar.
     * @return Probability-constrained result of transforming
     * the free scalar.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T prob_constrain(const T x) {
      using stan::math::inv_logit;
      return inv_logit(x);
    }

    /**
     * Return a probability value constrained to fall between 0 and 1
     * (inclusive) for the specified free scalar and increment the
     * specified log probability reference with the log absolute Jacobian
     * determinant of the transform.
     *
     * <p>The transform is as defined for <code>prob_constrain(T)</code>. 
     * The log absolute Jacobian determinant is
     *
     * <p>The log absolute Jacobian determinant is 
     *
     * <p>\f$\log | \frac{d}{dx} \mbox{logit}^{-1}(x) |\f$
     * <p>\f$\log ((\mbox{logit}^{-1}(x)) (1 - \mbox{logit}^{-1}(x))\f$
     * <p>\f$\log (\mbox{logit}^{-1}(x)) + \log (1 - \mbox{logit}^{-1}(x))\f$.
     *
     * @param x Free scalar.
     * @param lp Log probability reference.
     * @return Probability-constrained result of transforming
     * the free scalar.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T prob_constrain(const T x, T& lp) {
      using stan::math::inv_logit;
      using stan::math::log1m;
      T inv_logit_x = inv_logit(x);
      lp += log(inv_logit_x) + log1m(inv_logit_x);
      return inv_logit_x;
    }

  }

}
#endif
