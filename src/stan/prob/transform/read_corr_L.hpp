#ifndef __STAN__PROB__TRANSFORM_READ_CORR_L_HPP__
#define __STAN__PROB__TRANSFORM_READ_CORR_L_HPP__

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
     * Return the Cholesky factor of the correlation matrix of the
     * specified dimensionality corresponding to the specified
     * canonical partial correlations.
     * 
     * It is generally better to work with the Cholesky factor rather
     * than the correlation matrix itself when the determinant,
     * inverse, etc. of the correlation matrix is needed for some
     * statistical calculation.
     *
     * <p>See <code>read_corr_matrix(Array,size_t,T)</code>
     * for more information.
     *
     * @param CPCs The (K choose 2) canonical partial correlations in
     * (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @return Cholesky factor of correlation matrix for specified
     * canonical partial correlations.

     * @tparam T Type of underlying scalar.  
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_L(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs, // on (-1,1)
                const size_t K) {
      Eigen::Array<T,Eigen::Dynamic,1> temp;         
      Eigen::Array<T,Eigen::Dynamic,1> acc(K-1);  
      acc.setOnes();
      // Cholesky factor of correlation matrix
      Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic> L(K,K); 
      L.setZero();

      size_t position = 0;
      size_t pull = K - 1;

      L(0,0) = 1.0;
      L.col(0).tail(pull) = temp = CPCs.head(pull);
      acc.tail(pull) = 1.0 - temp.square();
      for(size_t i = 1; i < (K - 1); i++) {
        position += pull;
        pull--;
        temp = CPCs.segment(position, pull);
        L(i,i) = sqrt(acc(i-1));
        L.col(i).tail(pull) = temp * acc.tail(pull).sqrt();
        acc.tail(pull) *= 1.0 - temp.square();
      }
      L(K-1,K-1) = sqrt(acc(K-2));
      return L.matrix();
    }

 /**
     * Return the Cholesky factor of the correlation matrix of the
     * specified dimensionality corresponding to the specified
     * canonical partial correlations, incrementing the specified
     * scalar reference with the log absolute determinant of the
     * Jacobian of the transformation.
     *
     * <p>The implementation is Ben Goodrich's Cholesky
     * factor-based approach to the C-vine method of:
     * 
     * <ul><li> Daniel Lewandowski, Dorota Kurowicka, and Harry Joe,
     * Generating random correlation matrices based on vines and
     * extended onion method Journal of Multivariate Analysis 100
     * (2009) 1989–2001 </li></ul>
     *
     * // FIXME: explain which CPCs we're dealing with
     * 
     * @param CPCs The (K choose 2) canonical partial correlations in
     * (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @param log_prob Reference to variable to increment with the log
     * Jacobian determinant.
     * @return Cholesky factor of correlation matrix for specified
     * partial correlations.
     * @tparam T Type of underlying scalar.  
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_L(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                const size_t K,
                T& log_prob) {

      size_t k = 0; 
      size_t i = 0;
      T log_1cpc2;
      double lead = K - 2.0; 
      // no need to abs() because this Jacobian determinant 
      // is strictly positive (and triangular)
      // skip last row (odd indexing) because it adds nothing by design
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      for (size_type j = 0; 
           j < (CPCs.rows() - 1);
           ++j) {
        using stan::math::log1m;
        using stan::math::square;
        log_1cpc2 = log1m(square(CPCs[j]));
        // derivative of correlation wrt CPC
        log_prob += lead / 2.0 * log_1cpc2; 
        i++;
        if (i > K) {
          k++;
          i = k + 1;
          lead = K - k - 1.0;
        }
      }
      return read_corr_L(CPCs, K);
    }

  }

}
#endif
