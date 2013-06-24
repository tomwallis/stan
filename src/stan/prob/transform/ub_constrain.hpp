#ifndef __STAN__PROB__TRANSFORM_UB_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_UB_CONSTRAIN_HPP__

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
#include <stan/prob/transform/identity_constrain.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

namespace stan {
  
  namespace prob {

    /**
     * Return the upper-bounded value for the specified unconstrained
     * scalar and upper bound.
     *
     * <p>The transform is
     *
     * <p>\f$f(x) = U - \exp(x)\f$
     *
     * <p>where \f$U\f$ is the upper bound.  
     * 
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_constrain(x)</code>.
     * 
     * @param x Free scalar.
     * @param ub Upper bound.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TU>::type
    ub_constrain(const T x, const TU ub) {
      if (ub == std::numeric_limits<double>::infinity())
        return identity_constrain(x);
      return ub - exp(x);
    }

    /**
     * Return the upper-bounded value for the specified unconstrained
     * scalar and upper bound and increment the specified log
     * probability reference with the log absolute Jacobian
     * determinant of the transform.
     *
     * <p>The transform is as specified for
     * <code>ub_constrain(T,double)</code>.  The log absolute Jacobian
     * determinant is
     *
     * <p>\f$ \log | \frac{d}{dx} -\mbox{exp}(x) + U | 
     *     = \log | -\mbox{exp}(x) + 0 | = x\f$.
     *
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_constrain(x,lp)</code>.
     *
     * @param x Free scalar.
     * @param ub Upper bound.
     * @param lp Log probability reference.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TU>::type
    ub_constrain(const T x, const TU ub, T& lp) {
      if (ub == std::numeric_limits<double>::infinity())
        return identity_constrain(x,lp);
      lp += x;
      return ub - exp(x);
    }

  }

}
#endif
