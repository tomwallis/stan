#ifndef __STAN__PROB__TRANSFORM_IDENTITY_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_IDENTITY_CONSTRAIN_HPP__

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
     * Returns the result of applying the identity constraint
     * transform to the input.
     *
     * <p>This method is effectively a no-op and is mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param x Free scalar.
     * @return Transformed input.
     *
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline 
    T identity_constrain(T x) {
      return x;
    }

    /**
     * Returns the result of applying the identity constraint
     * transform to the input and increments the log probability
     * reference with the log absolute Jacobian determinant.
     *
     * <p>This method is effectively a no-op and mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param x Free scalar.
     * @param lp Reference to log probability.
     * @return Transformed input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline 
    T identity_constrain(const T x, T& /*lp*/) {
      return x;
    }

  }

}
#endif
