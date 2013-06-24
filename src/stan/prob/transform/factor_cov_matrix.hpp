#ifndef __STAN__PROB__TRANSFORM_FACTOR_COV_MATRIX_HPP__
#define __STAN__PROB__TRANSFORM_FACTOR_COV_MATRIX_HPP__

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
     * This function is intended to make starting values, given a
     * covariance matrix Sigma
     *
     * The transformations are hard coded as log for standard
     * deviations and Fisher transformations (atanh()) of CPCs
     *
     * @param CPCs fill this unbounded
     * @param sds fill this unbounded
     * @param Sigma covariance matrix
     * @return false if any of the diagonals of Sigma are 0
     */
    template<typename T>
    bool
    factor_cov_matrix(Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
              Eigen::Array<T,Eigen::Dynamic,1>& sds, 
              const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {

      size_t K = sds.rows();

      sds = Sigma.diagonal().array();
      if( (sds <= 0.0).any() ) return false;
      sds = sds.sqrt();

      Eigen::DiagonalMatrix<T,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();
      sds = sds.log(); // now unbounded

      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> R = D * Sigma * D;
      // to hopefully prevent pivoting due to floating point error
      R.diagonal().setOnes(); 
      Eigen::LDLT<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > ldlt;
      ldlt = R.ldlt();
      if (!ldlt.isPositive()) 
        return false;
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> U = ldlt.matrixU();

      size_t position = 0;
      size_t pull = K - 1;

      Eigen::Array<T,1,Eigen::Dynamic> temp = U.row(0).tail(pull);

      CPCs.head(pull) = temp;

      Eigen::Array<T,Eigen::Dynamic,1> acc(K);
      acc(0) = -0.0;
      acc.tail(pull) = 1.0 - temp.square();
      for(size_t i = 1; i < (K - 1); i++) {
        position += pull;
        pull--;
        temp = U.row(i).tail(pull);
        temp /= sqrt(acc.tail(pull) / acc(i));
        CPCs.segment(position, pull) = temp;
        acc.tail(pull) *= 1.0 - temp.square();
      }
      CPCs = 0.5 * ( (1.0 + CPCs) / (1.0 - CPCs) ).log(); // now unbounded
      return true;
    }

  }       
 }

#endif
