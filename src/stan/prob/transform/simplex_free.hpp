#ifndef __STAN__PROB__TRANSFORM_SIMPLEX_FREE_HPP__
#define __STAN__PROB__TRANSFORM_SIMPLEX_FREE_HPP__

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
     * Return an unconstrained vector that when transformed produces
     * the specified simplex.  It applies to a simplex of dimensionality
     * K and produces an unconstrained vector of dimensionality (K-1).
     *
     * <p>The simplex transform is defined through a centered
     * stick-breaking process.
     * 
     * @param x Simplex of dimensionality K.
     * @return Free vector of dimensionality (K-1) that transfroms to
     * the simplex.
     * @tparam T Type of scalar.
     * @throw std::domain_error if x is not a valid simplex
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    simplex_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      using stan::math::logit;
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      stan::math::check_simplex("stan::prob::simplex_free(%1%)", x, "Simplex variable");
      int Km1 = x.size() - 1;
      Eigen::Matrix<T,Eigen::Dynamic,1> y(Km1);
      T stick_len(x(Km1));
      for (size_type k = Km1; --k >= 0; ) {
        stick_len += x(k);
        T z_k(x(k) / stick_len);
        y(k) = logit(z_k) + log(Km1 - k); 
        // log(Km-k) = logit(1.0 / (Km1 + 1 - k));
      }
      return y;
    }

  }

}
#endif
