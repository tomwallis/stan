#ifndef __STAN__PROB__TRANSFORM_UB_FREE_HPP__
#define __STAN__PROB__TRANSFORM_UB_FREE_HPP__

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
     * Return the free scalar that corresponds to the specified
     * upper-bounded value with respect to the specified upper bound.
     *
     * <p>The transform is the reverse of the
     * <code>ub_constrain(T,double)</code> transform,
     *
     * <p>\f$f^{-1}(y) = \log -(y - U)\f$
     *
     * <p>where \f$U\f$ is the upper bound.
     *
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_free(y)</code>.
     *
     * @param y Upper-bounded scalar.
     * @param ub Upper bound.
     * @return Free scalar corresponding to upper-bounded scalar.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     * @throw std::invalid_argument if y is greater than the upper
     * bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TU>::type
    ub_free(const T y, const TU ub) {
      if (ub == std::numeric_limits<double>::infinity())
        return identity_free(y);
      stan::math::check_less_or_equal("stan::prob::ub_free(%1%)",
                                      y, ub, "Upper bounded variable");
      return log(ub - y);
    }
  }

}
#endif
