#ifndef __STAN__PROB__TRANSFORM_PROB_FREE_HPP__
#define __STAN__PROB__TRANSFORM_PROB_FREE_HPP__

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
     * Return the free scalar that when transformed to a probability
     * produces the specified scalar.
     *
     * <p>The function that reverses the constraining transform
     * specified in <code>prob_constrain(T)</code> is the logit
     * function,
     *
     * <p>\f$f^{-1}(y) = \mbox{logit}(y) = \frac{1 - y}{y}\f$.
     * 
     * @param y Scalar input.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is less than 0 or greater than 1.
     */
    template <typename T>
    inline
    T prob_free(const T y) {
      using stan::math::logit;
      stan::math::check_bounded("stan::prob::prob_free(%1%)",
                                y, 0, 1, "Probability variable");
      return logit(y);
    }

  }

}
#endif
