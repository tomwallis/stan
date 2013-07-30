#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP__

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
//#include <boost/utility/enable_if.hpp>

//#include <stan/agrad.hpp>
//#include <stan/math.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the normal density for the specified scalar(s) given
     * the specified mean(s) and deviation(s). y, mu, or sigma can
     * each be either a scalar or a std::vector. Any vector inputs
     * must be the same length.
     *
     * <p>The result log probability is defined to be the sum of the
     * log probabilities for each observation/mean/deviation triple.
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location parameter(s)
     * for the normal distribution.
     * @param sigma (Sequence of) scale parameters for the normal
     * distribution.
     * @return The log of the product of the densities.
     * @throw std::domain_error if the scale is not positive.
     * @tparam T_y Underlying type of scalar in sequence.
     * @tparam T_loc Type of location parameter.
     */
    template <bool propto, 
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma);

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma);
    
    /**
     * Calculates the normal cumulative distribution function for the given
     * variate, location, and scale.
     * 
     * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
     * 
     * Errors are configured by policy.  All variables must be finite
     * and the scale must be strictly greater than zero.
     * 
     * @param y A scalar variate.
     * @param mu The location of the normal distribution.
     * @param sigma The scale of the normal distriubtion
     * @return The unit normal cdf evaluated at the specified arguments.
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     * @tparam Policy Error-handling policy.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma);

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma);

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma);

    template <class RNG>
    inline double
    normal_rng(const double mu,
               const double sigma,
               RNG& rng);
  }
}
#endif
