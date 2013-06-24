#ifndef __STAN__PROB__TRANSFORM_COV_MATRIX_LKJ_FREE_HPP__
#define __STAN__PROB__TRANSFORM_COV_MATRIX_LKJ_FREE_HPP__

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
#include <stan/prob/transform/factor_cov_matrix.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

namespace stan {
  
  namespace prob {

    /**
     * Return the vector of unconstrained partial correlations and
     * deviations that transform to the specified covariance matrix.
     *
     * <p>The constraining transform is defined as for
     * <code>cov_matrix_constrain(Matrix,size_t)</code>.  The
     * inverse first factors out the deviations, then applies the
     * freeing transfrom of <code>corr_matrix_free(Matrix&)</code>.
     *
     * @param y Covariance matrix to free.
     * @return Vector of unconstrained values that transforms to the
     * specified covariance matrix.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the correlation matrix has no
     *    elements or is not a square matrix.
     * @throw std::runtime_error if the correlation matrix cannot be
     *    factorized by factor_cov_matrix()
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    cov_matrix_free_lkj(
            const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y) {
      typedef typename
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;
      size_type k = y.rows();
      if (y.cols() != k)
        throw std::domain_error("y is not a square matrix");
      if (k == 0)
        throw std::domain_error("y has no elements");
      size_type k_choose_2 = (k * (k-1)) / 2;
      Eigen::Array<T,Eigen::Dynamic,1> cpcs(k_choose_2);
      Eigen::Array<T,Eigen::Dynamic,1> sds(k);
      bool successful = factor_cov_matrix(cpcs,sds,y);
      if (!successful)
        throw std::runtime_error ("factor_cov_matrix failed on y");
      Eigen::Matrix<T,Eigen::Dynamic,1> x(k_choose_2 + k);
      size_type pos = 0;
      for (size_type i = 0; i < k_choose_2; ++i)
        x[pos++] = cpcs[i];
      for (size_type i = 0; i < k; ++i)
        x[pos++] = sds[i];
      return x;
    }

  }

}
#endif
