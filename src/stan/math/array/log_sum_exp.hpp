#ifndef __STAN__MATH__ARRAY__LOG_SUM_EXP_HPP__
#define __STAN__MATH__ARRAY__LOG_SUM_EXP_HPP__

#include <stan/math/scalar/log1p.hpp>
#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <limits>

namespace stan {
  namespace math {

    /**
     * Return the log of the sum of the exponentiated values of the specified
     * sequence of values.
     *
     * The function is defined as follows to prevent overflow in exponential
     * calculations.
     *
     * \f$\log \sum_{n=1}^N \exp(x_n) = \max(x) + \log \sum_{n=1}^N \exp(x_n - \max(x))\f$.
     * 
     * @param[in] x array of specified values
     * @return The log of the sum of the exponentiated vector values.
     */
    template <typename T>
    T log_sum_exp(const std::vector<T>& x) {
      using std::numeric_limits;
      using std::log;
      using std::exp;
      T max = -numeric_limits<T>::infinity();
      for (size_t ii = 0; ii < x.size(); ii++) 
        if (x[ii] > max) 
          max = x[ii];
            
      T sum = 0.0;
      for (size_t ii = 0; ii < x.size(); ii++) 
        if (x[ii] != -numeric_limits<double>::infinity()) 
          sum += exp(x[ii] - max);
          
      return max + log(sum);
    }


  }
}

#endif
