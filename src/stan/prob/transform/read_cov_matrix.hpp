#ifndef __STAN__PROB__TRANSFORM_READ_COV_MATRIX_HPP__
#define __STAN__PROB__TRANSFORM_READ_COV_MATRIX_HPP__

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
#include <stan/prob/transform/read_corr_L.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

namespace stan {
  
  namespace prob {

    /** 
     * A generally worse alternative to call prior to evaluating the
     * density of an elliptical distribution
     *
     * @param CPCs on (-1,1)
     * @param sds on (0,inf)
     * @param log_prob the log probability value to increment with the Jacobian
     * @return Covariance matrix for specified partial correlations.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_cov_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                    const Eigen::Array<T,Eigen::Dynamic,1>& sds, 
                    T& log_prob) {

      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L 
        = read_cov_L(CPCs, sds, log_prob);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }

    /** 
     *
     * Builds a covariance matrix from CPCs and standard deviations
     *
     * @param CPCs in (-1,1)
     * @param sds in (0,inf)
     */
    template<typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_cov_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs, 
                    const Eigen::Array<T,Eigen::Dynamic,1>& sds) {

      size_t K = sds.rows();
      Eigen::DiagonalMatrix<T,Eigen::Dynamic> D(K);
      D.diagonal() = sds;
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L 
        = D * read_corr_L(CPCs, K);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }

  }

}
#endif
