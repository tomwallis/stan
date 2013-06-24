#ifndef __STAN__PROB__TRANSFORM_LB_FREE_HPP__
#define __STAN__PROB__TRANSFORM_LB_FREE_HPP__

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
#include <stan/prob/transform/identity_free.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

namespace stan {
  
  namespace prob {

  /**
     * Return the unconstrained value that produces the specified
     * lower-bound constrained value.
     *
     * If the lower bound is negative infinity, it is ignored and
     * the function reduces to <code>identity_free(y)</code>.
     * 
     * @param y Input scalar.
     * @param lb Lower bound.
     * @return Unconstrained value that produces the input when
     * constrained.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     * @throw std::domain_error if y is lower than the lower bound.
     */
    template <typename T, typename TL>
    inline
    typename boost::math::tools::promote_args<T,TL>::type
    lb_free(const T y, const TL lb) {
      if (lb == -std::numeric_limits<double>::infinity())
        return identity_free(y);
      stan::math::check_greater_or_equal("stan::prob::lb_free(%1%)",
                                         y, lb, "Lower bounded variable");
      return log(y - lb);
    }
  }

}
#endif
