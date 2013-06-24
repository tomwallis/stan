#ifndef __STAN__PROB__TRANSFORM_MAKE_NU_HPP__
#define __STAN__PROB__TRANSFORM_MAKE_NU_HPP__

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
     * This function calculates the degrees of freedom for the t
     * distribution that corresponds to the shape parameter in the
     * Lewandowski et. al. distribution 
     *
     * @param eta hyperparameter on (0,inf), eta = 1 <-> correlation
     * matrix is uniform
     * @param K number of variables in covariance matrix
     */
    template<typename T>
    const Eigen::Array<T,Eigen::Dynamic,1>
    make_nu(const T eta, const size_t K) {
  
      Eigen::Array<T,Eigen::Dynamic,1> nu(K * (K - 1) / 2);
  
      T alpha = eta + (K - 2.0) / 2.0; // from Lewandowski et. al.

      // Best (1978) implies nu = 2 * alpha for the dof in a t 
      // distribution that generates a beta variate on (-1,1)
      T alpha2 = 2.0 * alpha; 

      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      for (size_type j = 0; j < (K - 1); j++) {
        nu(j) = alpha2;
      }
      size_t counter = K - 1;
      for (size_type i = 1; i < (K - 1); i++) {
        alpha -= 0.5;
        alpha2 = 2.0 * alpha;
        for (size_type j = i + 1; j < K; j++) {
          nu(counter) = alpha2;
          counter++;
        }
      }
      return nu;
    }

  }

}
#endif
