#ifndef __STAN__PROB__TRANSFORM_SIMPLEX_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_SIMPLEX_CONSTRAIN_HPP__

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
     * Return the simplex corresponding to the specified free vector.  
     * A simplex is a vector containing values greater than or equal
     * to 0 that sum to 1.  A vector with (K-1) unconstrained values
     * will produce a simplex of size K.
     *
     * The transform is based on a centered stick-breaking process.
     *
     * @param y Free vector input of dimensionality K - 1.
     * @return Simplex of dimensionality K.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    simplex_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {
      // cut & paste simplex_constrain(Eigen::Matrix,T) w/o Jacobian
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      using stan::math::logit;
      using stan::math::inv_logit;
      using stan::math::log1m;
      int Km1 = y.size();
      Eigen::Matrix<T,Eigen::Dynamic,1> x(Km1 + 1);
      T stick_len(1.0);
      for (size_type k = 0; k < Km1; ++k) {
        T z_k(inv_logit(y(k) - log(Km1 - k))); 
        x(k) = stick_len * z_k;
        stick_len -= x(k); 
      }
      x(Km1) = stick_len;
      return x;
    }

    /**
     * Return the simplex corresponding to the specified free vector
     * and increment the specified log probability reference with 
     * the log absolute Jacobian determinant of the transform. 
     *
     * The simplex transform is defined through a centered
     * stick-breaking process.
     * 
     * @param y Free vector input of dimensionality K - 1.
     * @param lp Log probability reference to increment.
     * @return Simplex of dimensionality K.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    simplex_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y, 
                      T& lp) {
      using stan::math::logit;
      using stan::math::inv_logit;
      using stan::math::log1p_exp;
      using stan::math::log1m;
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      int Km1 = y.size(); // K = Km1 + 1
      Eigen::Matrix<T,Eigen::Dynamic,1> x(Km1 + 1);
      T stick_len(1.0);
      for (size_type k = 0; k < Km1; ++k) {
        double eq_share = -log(Km1 - k); // = logit(1.0/(Km1 + 1 - k));
        T adj_y_k(y(k) + eq_share);
        T z_k(inv_logit(adj_y_k));
        x(k) = stick_len * z_k;
        lp += log(stick_len);
        lp -= log1p_exp(-adj_y_k);
        lp -= log1p_exp(adj_y_k);
        stick_len -= x(k); // equivalently *= (1 - z_k);
      }
      x(Km1) = stick_len; // no Jacobian contrib for last dim
      return x;
    }

  }

}
#endif
