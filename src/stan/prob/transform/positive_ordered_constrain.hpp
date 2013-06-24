#ifndef __STAN__PROB__TRANSFORM_POSITIVE_ORDERED_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_POSITIVE_ORDERED_CONSTRAIN_HPP__

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
     * Return an increasing positive ordered vector derived from the specified
     * free vector.  The returned constrained vector will have the
     * same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @return Positive, increasing ordered vector.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    positive_ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      size_type k = x.size();
      Eigen::Matrix<T,Eigen::Dynamic,1> y(k);
      if (k == 0)
        return y;
      y[0] = exp(x[0]);
      for (size_type i = 1; 
           i < k; 
           ++i)
        y[i] = y[i-1] + exp(x[i]);
      return y;
    }

    /**
     * Return a positive valued, increasing positive ordered vector derived
     * from the specified free vector and increment the specified log
     * probability reference with the log absolute Jacobian determinant
     * of the transform.  The returned constrained vector
     * will have the same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @param lp Log probability reference.
     * @return Positive, increasing ordered vector. 
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1> 
    positive_ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, T& lp) {
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      for (size_type i = 0; i < x.size(); ++i)
        lp += x(i);
      return positive_ordered_constrain(x);
    }

  }

}
#endif
