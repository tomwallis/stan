#ifndef __STAN__PROB__TRANSFORM_UNIT_VECTOR_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_UNIT_VECTOR_CONSTRAIN_HPP__

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
     * Return the unit length vector corresponding to the free vector y.
     * The free vector contains K-1 spherical coordinates.
     *
     * @param Vector of K - 1 spherical coordinates
     * @return Unit length vector of dimension K
     * @tparam T Scalar type.
     **/
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    unit_vector_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      int Km1 = y.size();
      Eigen::Matrix<T,Eigen::Dynamic,1> x(Km1 + 1);
      x(0) = 1.0;
      const T half_pi = T(M_PI/2.0);
      for (size_type k = 1; k <= Km1; ++k) {
        T yk_1 = y(k-1) + half_pi;
        T sin_yk_1 = sin(yk_1);
        x(k) = x(k-1)*sin_yk_1; 
        x(k-1) *= cos(yk_1);
      }
      return x;
    }

    /**
     * Return the unit length vector corresponding to the free vector y.
     * The free vector contains K-1 spherical coordinates.
     *
     * @param Vector of K - 1 spherical coordinates
     * @return Unit length vector of dimension K
     * @param lp Log probability reference to increment.
     * @tparam T Scalar type.
     **/
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    unit_vector_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& y, T &lp) {
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      int Km1 = y.size();
      Eigen::Matrix<T,Eigen::Dynamic,1> x(Km1 + 1);
      x(0) = 1.0;
      const T half_pi = T(M_PI/2.0);
      for (size_type k = 1; k <= Km1; ++k) {
        T yk_1 = y(k-1) + half_pi;
        T sin_yk_1 = sin(yk_1);
        x(k) = x(k-1)*sin_yk_1; 
        x(k-1) *= cos(yk_1);
        if (k < Km1)
          lp += (Km1 - k)*log(fabs(sin_yk_1));
      }
      return x;
    }

  }

}
#endif
