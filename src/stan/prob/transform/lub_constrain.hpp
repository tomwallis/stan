#ifndef __STAN__PROB__TRANSFORM_LUB_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_LUB_CONSTRAIN_HPP__

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
#include <stan/prob/transform/ub_constrain.hpp>
#include <stan/prob/transform/lb_constrain.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

namespace stan {
  
  namespace prob {

    /**
     * Return the lower- and upper-bounded scalar derived by
     * transforming the specified free scalar given the specified
     * lower and upper bounds.
     *
     * <p>The transform is the transformed and scaled inverse logit,
     *
     * <p>\f$f(x) = L + (U - L) \mbox{logit}^{-1}(x)\f$
     *
     * If the lower bound is negative infinity and upper bound finite,
     * this function reduces to <code>ub_constrain(x,ub)</code>.  If
     * the upper bound is positive infinity and the lower bound
     * finite, this function reduces to
     * <code>lb_constrain(x,lb)</code>.  If the upper bound is
     * positive infinity and the lower bound negative infinity, 
     * this function reduces to <code>identity_constrain(x)</code>.
     * 
     * @param x Free scalar to transform.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @return Lower- and upper-bounded scalar derived from transforming
     * the free scalar.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     * @tparam TU Type of upper bound.
     * @throw std::domain_error if ub <= lb
     */
    template <typename T, typename TL, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TL,TU>::type
    lub_constrain(const T x, TL lb, TU ub) {
      stan::math::validate_less(lb,ub,"lb","ub","lub_constrain/3");

      if (lb == -std::numeric_limits<double>::infinity())
        return ub_constrain(x,ub);
      if (ub == std::numeric_limits<double>::infinity())
        return lb_constrain(x,lb);

      T inv_logit_x;
      if (x > 0) {
        T exp_minus_x = exp(-x);
        inv_logit_x = 1.0 / (1.0 + exp_minus_x);
        // Prevent x from reaching one unless it really really should.
        if ((x < std::numeric_limits<double>::infinity()) 
            && (inv_logit_x == 1))
            inv_logit_x = 1 - 1e-15;
      } else {
        T exp_x = exp(x);
        inv_logit_x = 1.0 - 1.0 / (1.0 + exp_x);
        // Prevent x from reaching zero unless it really really should.
        if ((x > -std::numeric_limits<double>::infinity()) 
            && (inv_logit_x== 0))
            inv_logit_x = 1e-15;
      }
      return lb + (ub - lb) * inv_logit_x;
    }

    /**
     * Return the lower- and upper-bounded scalar derived by
     * transforming the specified free scalar given the specified
     * lower and upper bounds and increment the specified log
     * probability with the log absolute Jacobian determinant.
     *
     * <p>The transform is as defined in
     * <code>lub_constrain(T,double,double)</code>.  The log absolute
     * Jacobian determinant is given by
     * 
     * <p>\f$\log \left| \frac{d}{dx} \left(
     *                L + (U-L) \mbox{logit}^{-1}(x) \right) 
     *            \right|\f$
     *
     * <p>\f$ {} = \log |
     *         (U-L)
     *         \, (\mbox{logit}^{-1}(x)) 
     *         \, (1 - \mbox{logit}^{-1}(x)) |\f$
     *
     * <p>\f$ {} = \log (U - L) + \log (\mbox{logit}^{-1}(x)) 
     *                          + \log (1 - \mbox{logit}^{-1}(x))\f$
     *
     * <p>If the lower bound is negative infinity and upper bound finite,
     * this function reduces to <code>ub_constrain(x,ub,lp)</code>.  If
     * the upper bound is positive infinity and the lower bound
     * finite, this function reduces to
     * <code>lb_constrain(x,lb,lp)</code>.  If the upper bound is
     * positive infinity and the lower bound negative infinity, 
     * this function reduces to <code>identity_constrain(x,lp)</code>.
     *
     * @param x Free scalar to transform.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @param lp Log probability scalar reference.
     * @return Lower- and upper-bounded scalar derived from transforming
     * the free scalar.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     * @tparam TU Type of upper bound.
     * @throw std::domain_error if ub <= lb
     */
    template <typename T, typename TL, typename TU>
    typename boost::math::tools::promote_args<T,TL,TU>::type
    lub_constrain(const T x, const TL lb, const TU ub, T& lp) {
      if (!(lb < ub)) {
        std::stringstream s;
        s << "domain error in lub_constrain;  lower bound = " << lb
          << " must be strictly less than upper bound = " << ub;
        throw std::domain_error(s.str());
      }
      if (lb == -std::numeric_limits<double>::infinity())
        return ub_constrain(x,ub,lp);
      if (ub == std::numeric_limits<double>::infinity())
        return lb_constrain(x,lb,lp);
      T inv_logit_x;
      if (x > 0) {
        T exp_minus_x = exp(-x);
        inv_logit_x = 1.0 / (1.0 + exp_minus_x);
        lp += log(ub - lb) - x - 2 * log1p(exp_minus_x);
        // Prevent x from reaching one unless it really really should.
        if ((x < std::numeric_limits<double>::infinity()) 
            && (inv_logit_x == 1))
            inv_logit_x = 1 - 1e-15;
      } else {
        T exp_x = exp(x);
        inv_logit_x = 1.0 - 1.0 / (1.0 + exp_x);
        lp += log(ub - lb) + x - 2 * log1p(exp_x);
        // Prevent x from reaching zero unless it really really should.
        if ((x > -std::numeric_limits<double>::infinity()) 
            && (inv_logit_x== 0))
            inv_logit_x = 1e-15;
      }
      return lb + (ub - lb) * inv_logit_x;
    }

  }

}
#endif
