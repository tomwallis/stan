// Arguments: Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/discrete/neg_binomial.hpp>
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/log1m.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsNegBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 2.0;          // alpha
    param[2] = 1.5;          // beta
    parameters.push_back(param);
    log_prob.push_back(-7.786663); // expected log_prob

    param[0] = 100;          // n
    param[1] = 3.0;          // alpha
    param[2] = 3.5;          // beta
    parameters.push_back(param);
    log_prob.push_back(-142.6147); // expected log_prob

    param[0] = 13;
    param[1] = 1e11; // alpha > 1e10, causes redux to Poisson
    param[2] = 1e10; // equiv to Poisson(1e11/1e10) = Poisson(10)
    parameters.push_back(param);
    log_prob.push_back(-2.618558); // log poisson(13|10)
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    // alpha
    index.push_back(1U);
    value.push_back(0);
    
    // beta
    index.push_back(2U);
    value.push_back(0);
  }

  double num_params() {
    return 3;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0); //n
    lb.push_back(1); //alpha
    lb.push_back(1.0e-300); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(-1.0216513); //n for valid values 1
    lb1.push_back(-0.75394330); //n for valid values 2
    lb1.push_back(-10.0); //n for valid values 3
    lb2.push_back(-9.6737327); //alpha for valid values 1
    lb2.push_back(-150.65906); //alpha for valid values 2
    lb2.push_back(-299.33606); //alpha for valid values 3
    lb3.push_back(-1379.1532); //beta for valid values 1
    lb3.push_back(-2063.7796); //beta for valid values 2
    lb3.push_back(numeric_limits<float>::quiet_NaN()); //beta for valid values 3

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<int>::infinity()); //n
    ub.push_back(numeric_limits<double>::infinity()); //alpha
    ub.push_back(1.0); //beta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
   
    ub1.push_back(-1.0216513); //n for valid values 1
    ub1.push_back(-0.75394330); //n for valid values 2
    ub1.push_back(-10.0); //n for valid values 3
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 3
    ub3.push_back(-5.9198710); //beta for valid values 1
    ub3.push_back(-62.847213); //beta for valid values 2
    ub3.push_back(-9.9999997e10); //beta for valid values 3

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }

  template <class T_n, class T_shape, class T_inv_scale,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, 
            typename T9>
  typename stan::return_type<T_shape,T_inv_scale>::type 
  log_prob(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
     const T3&, const T4&, const T5&, 
     const T6&, const T7&, const T8&, 
     const T9&) {
    return stan::prob::neg_binomial_log(n, alpha, beta);
  }

  template <bool propto, 
      class T_n, class T_shape, class T_inv_scale,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, 
            typename T9>
  typename stan::return_type<T_shape,T_inv_scale>::type 
  log_prob(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
     const T3&, const T4&, const T5&, 
     const T6&, const T7&, const T8&, 
     const T9&) {
    return stan::prob::neg_binomial_log<propto>(n, alpha, beta);
  }
  

  template <class T_n, class T_shape, class T_inv_scale,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, 
            typename T9>
  var log_prob_function(const T_n& n, const T_shape& alpha, const T_inv_scale& beta,
      const T3&, const T4&, const T5&, 
      const T6&, const T7&, const T8&, 
      const T9&) {
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log1m;
    using stan::math::multiply_log;
    using stan::prob::include_summand;

    var logp(0);
    // Special case where negative binomial reduces to Poisson
    if (alpha > 1e10) {
      if (include_summand<true>::value)
        logp -= lgamma(n + 1.0);
      if (include_summand<true,T_shape>::value ||
          include_summand<true,T_inv_scale>::value) {
        typename stan::return_type<T_shape, T_inv_scale>::type lambda;
        lambda = alpha / beta;
        logp += multiply_log(n, lambda) - lambda;
      }
      return logp;
    }
    // More typical cases
    if (include_summand<true,T_shape>::value)
      if (n != 0)
        logp += binomial_coefficient_log<typename stan::scalar_type<T_shape>::type>
          (n + alpha - 1.0, n);
    if (include_summand<true,T_shape,T_inv_scale>::value)
      logp += -n * log1p(beta) 
        + alpha * log(beta / (1 + beta));
    return logp;
  }
};

