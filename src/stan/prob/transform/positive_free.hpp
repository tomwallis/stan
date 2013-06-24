#ifndef __STAN__PROB__TRANSFORM_POSITIVE_FREE_HPP__
#define __STAN__PROB__TRANSFORM_POSITIVE_FREE_HPP__

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
     * Return the unconstrained value corresponding to the specified
     * positive-constrained value.  
     *
     * <p>The transform is the inverse of the transform \f$f\f$ applied by
     * <code>positive_constrain(T)</code>, namely
     *
     * <p>\f$f^{-1}(x) = \log(x)\f$.
     * 
     * <p>The input is validated using <code>stan::math::check_positive()</code>.
     * 
     * @param y Input scalar.
     * @return Unconstrained value that produces the input when constrained.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the variable is negative.
     */
    template <typename T>
    inline
    T positive_free(const T y) {
      stan::math::check_positive("stan::prob::positive_free(%1%)", y, "Positive variable");
      return log(y);
    }

  }

}
#endif
