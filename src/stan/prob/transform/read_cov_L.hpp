#ifndef __STAN__PROB__TRANSFORM_READ_COV_L_HPP__
#define __STAN__PROB__TRANSFORM_READ_COV_L_HPP__

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
     * This is the function that should be called prior to evaluating
     * the density of any elliptical distribution
     *
     * @param CPCs on (-1,1)
     * @param sds on (0,inf)
     * @param log_prob the log probability value to increment with the Jacobian
     * @return Cholesky factor of covariance matrix for specified
     * partial correlations.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_cov_L(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
               const Eigen::Array<T,Eigen::Dynamic,1>& sds, 
               T& log_prob) {
      size_t K = sds.rows();
      // adjust due to transformation from correlations to covariances
      log_prob += (sds.log().sum() + stan::math::LOG_2) * K;
      return sds.matrix().asDiagonal() * read_corr_L(CPCs, K, log_prob);
    }

  }

}
#endif
