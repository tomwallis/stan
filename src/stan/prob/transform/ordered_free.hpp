#ifndef __STAN__PROB__TRANSFORM_ORDERED_FREE_HPP__
#define __STAN__PROB__TRANSFORM_ORDERED_FREE_HPP__

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
     * Return the vector of unconstrained scalars that transform to
     * the specified positive ordered vector.
     *
     * <p>This function inverts the constraining operation defined in 
     * <code>ordered_constrain(Matrix)</code>,
     *
     * @param y Vector of positive, ordered scalars.
     * @return Free vector that transforms into the input vector.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is not a vector of positive,
     *   ordered scalars.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    ordered_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {
      stan::math::check_ordered("stan::prob::ordered_free(%1%)", 
                                y, "Ordered variable");
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      size_type k = y.size();
      Eigen::Matrix<T,Eigen::Dynamic,1> x(k);
      if (k == 0) 
        return x;
      x[0] = y[0];
      for (size_type i = 1; i < k; ++i)
        x[i] = log(y[i] - y[i-1]);
      return x;
    }

  }

}
#endif
