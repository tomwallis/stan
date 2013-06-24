#ifndef __STAN__PROB__TRANSFORM_COV_MATRIX_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_COV_MATRIX_CONSTRAIN_HPP__

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
     * Return the symmetric, positive-definite matrix of dimensions K
     * by K resulting from transforming the specified finite vector of
     * size K plus (K choose 2).
     *
     * <p>See <code>cov_matrix_free()</code> for the inverse transform.
     *
     * @param x The vector to convert to a covariance matrix.
     * @param K The number of rows and columns of the resulting
     * covariance matrix.
     * @throws std::domain_error if (x.size() != K + (K choose 2)).
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    cov_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                         typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type K) {
      using std::exp;
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L(K,K);
      if (x.size() != (K * (K + 1)) / 2) 
        throw std::domain_error("x.size() != K + (K choose 2)");
      int i = 0;
      for (size_type m = 0; m < K; ++m) {
        for (int n = 0; n < m; ++n)
          L(m,n) = x(i++);
        L(m,m) = exp(x(i++));
        for (size_type n = m + 1; n < K; ++n) 
          L(m,n) = 0.0;
      }
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> M = L * L.transpose();
      return L * L.transpose();
    }

    
    /**
     * Return the symmetric, positive-definite matrix of dimensions K
     * by K resulting from transforming the specified finite vector of
     * size K plus (K choose 2).
     *
     * <p>See <code>cov_matrix_free()</code> for the inverse transform.
     *
     * @param x The vector to convert to a covariance matrix.
     * @param K The dimensions of the resulting covariance matrix.
     * @param lp Reference
     * @throws std::domain_error if (x.size() != K + (K choose 2)).
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    cov_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                         typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type K,
                         T& lp) {
      using std::exp;
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      if (x.size() != (K * (K + 1)) / 2) 
        throw std::domain_error("x.size() != K + (K choose 2)");
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L(K,K);
      int i = 0;
      for (size_type m = 0; m < K; ++m) {
        for (size_type n = 0; n < m; ++n)
          L(m,n) = x(i++);
        L(m,m) = exp(x(i++));
        for (size_type n = m + 1; n < K; ++n) 
          L(m,n) = 0.0;
      }
      // Jacobian for complete transform, including exp() above
      lp += (K * stan::math::LOG_2); // needless constant; want propto
      for (int k = 0; k < K; ++k)
        lp += (K - k + 1) * log(L(k,k)); // only +1 because index from 0
      return L * L.transpose();
      // return tri_multiply_transpose(L); 
    }

  }

}
#endif
