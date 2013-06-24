#ifndef __STAN__PROB__TRANSFORM_CORR_FREE_HPP__
#define __STAN__PROB__TRANSFORM_CORR_FREE_HPP__

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
     * Return the unconstrained scalar that when transformed to
     * a valid correlation produces the specified value.
     *
     * <p>This function inverts the transform defined for
     * <code>corr_constrain(T)</code>, which is the inverse hyperbolic
     * tangent,
     *
     * <p>\f$ f^{-1}(y)
     *          = \mbox{atanh}\, y
     *          = \frac{1}{2} \log \frac{y + 1}{y - 1}\f$.
     *
     * @param y Correlation scalar input.
     * @return Free scalar that transforms to the specified input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_free(const T y) {
      stan::math::check_bounded("stan::prob::lub_free(%1%)",
                                y, -1, 1, "Correlation variable");
      return atanh(y);
    }

  }

}
#endif
