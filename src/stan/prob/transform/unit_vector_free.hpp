#ifndef __STAN__PROB__TRANSFORM_UNIT_VECTOR_FREE_HPP__
#define __STAN__PROB__TRANSFORM_UNIT_VECTOR_FREE_HPP__

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

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    unit_vector_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      stan::math::check_unit_vector("stan::prob::unit_vector_free(%1%)", x, "Unit vector variable");
      int Km1 = x.size() - 1;
      Eigen::Matrix<T,Eigen::Dynamic,1> y(Km1);
      T sumSq = x(Km1)*x(Km1);
      const T half_pi = T(M_PI/2.0);
      for (size_type k = Km1; --k >= 0; ) {
        y(k) = atan2(sqrt(sumSq),x(k)) - half_pi;
        sumSq += x(k)*x(k);
      }
      return y;
    }

  }

}
#endif
