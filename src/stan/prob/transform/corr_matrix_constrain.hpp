#ifndef __STAN__PROB__TRANSFORM_CORR_MATRIX_CONSTRAIN_HPP__
#define __STAN__PROB__TRANSFORM_CORR_MATRIX_CONSTRAIN_HPP__

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
#include <stan/prob/transform/corr_constrain.hpp>
#include <stan/prob/transform/read_corr_matrix.hpp>
#include <stan/prob/transform/factor_cov_matrix.hpp>

namespace stan {
  
  namespace prob {

    /**
     * Return the correlation matrix of the specified dimensionality
     * derived from the specified vector of unconstrained values.  The
     * input vector must be of length \f${k \choose 2} =
     * \frac{k(k-1)}{2}\f$.  The values in the input vector represent
     * unconstrained (partial) correlations among the dimensions.
     *
     * <p>The transform based on partial correlations is as specified
     * in
     *
     * <ul><li> Lewandowski, Daniel, Dorota Kurowicka, and Harry
     * Joe. 2009.  Generating random correlation matrices based on
     * vines and extended onion method.  <i>Journal of Multivariate
     * Analysis</i> <b>100</b>:1989–-2001.  </li></ul>
     *
     * <p>The free vector entries are first constrained to be
     * valid correlation values using <code>corr_constrain(T)</code>.
     * 
     * @param x Vector of unconstrained partial correlations.
     * @param k Dimensionality of returned correlation matrix.
     * @tparam T Type of scalar.
     * @throw std::invalid_argument if x is not a valid correlation
     * matrix.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    corr_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                 typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type k) {
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      size_type k_choose_2 = (k * (k - 1)) / 2;
      if (k_choose_2 != x.size())
        throw std::invalid_argument ("x is not a valid correlation matrix");
      Eigen::Array<T,Eigen::Dynamic,1> cpcs(k_choose_2);
      for (size_type i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[i]);
      return read_corr_matrix(cpcs,k); 
    }

    /**
     * Return the correlation matrix of the specified dimensionality
     * derived from the specified vector of unconstrained values.  The
     * input vector must be of length \f${k \choose 2} =
     * \frac{k(k-1)}{2}\f$.  The values in the input vector represent
     * unconstrained (partial) correlations among the dimensions.
     *
     * <p>The transform is as specified for
     * <code>corr_matrix_constrain(Matrix,size_t)</code>; the
     * paper it cites also defines the Jacobians for correlation inputs,
     * which are composed with the correlation constrained Jacobians 
     * defined in <code>corr_constrain(T,double)</code> for
     * this function.
     * 
     * @param x Vector of unconstrained partial correlations.
     * @param k Dimensionality of returned correlation matrix.
     * @param lp Log probability reference to increment.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> 
    corr_matrix_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                    typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type k,
                    T& lp) {
      typedef typename Eigen::Matrix<T,Eigen::Dynamic,1>::size_type size_type;
      size_type k_choose_2 = (k * (k - 1)) / 2;
      if (k_choose_2 != x.size())
        throw std::invalid_argument ("x is not a valid correlation matrix");
      Eigen::Array<T,Eigen::Dynamic,1> cpcs(k_choose_2);
      for (size_type i = 0; i < k_choose_2; ++i)
        cpcs[i] = corr_constrain(x[i],lp);
      return read_corr_matrix(cpcs,k,lp);
    }

  }

}
#endif
