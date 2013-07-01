// Arguments: Ints, Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>

#include <stan/math/functions/lbeta.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBetaBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(4);


    param[0] = 5;            // n
    param[1] = 20;           // N
    param[2] = 10.0;         // alpha
    param[3] = 25.0;         // beta
    parameters.push_back(param);
    log_prob.push_back(-1.854007); // expected log_prob

    param[0] = 25;           // n
    param[1] = 100;          // N
    param[2] = 30.0;         // alpha
    param[3] = 50.0;         // beta
    parameters.push_back(param);
    log_prob.push_back(-4.376696); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // n
    
    // N
    index.push_back(1U);
    value.push_back(-1);
    
    // alpha
    index.push_back(2U);
    value.push_back(0.0);
    
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-std::numeric_limits<double>::infinity());

    // beta
    index.push_back(3U);
    value.push_back(0.0);
    
    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(-std::numeric_limits<double>::infinity());
  }

  double num_params() {
    return 4;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0.0); //n
    lb.push_back(0.0); //N
    lb.push_back(1.0e-300); //alpha
    lb.push_back(1.0e-300); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
    std::vector<double> lb4;
   
    lb1.push_back(-5.206743); //n for valid values 1
    lb1.push_back(-28.320333); //n for valid values 2
    lb2.push_back(-numeric_limits<double>::infinity()); //N for valid values 1
    lb2.push_back(-numeric_limits<double>::infinity()); //N for valid values 2
    lb3.push_back(-696.6341); //alpha for valid values 1
    lb3.push_back(-705.1102); //alpha for valid values 2
    lb4.push_back(-702.0013); //beta for valid values 1
    lb4.push_back(-726.3024); //beta for valid values 2

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);
    lb.push_back(lb4);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<int>::infinity()); //n
    ub.push_back(numeric_limits<int>::infinity()); //N
    ub.push_back(numeric_limits<double>::infinity()); //alpha
    ub.push_back(numeric_limits<double>::infinity()); //beta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
    std::vector<double> ub4;
   
    ub1.push_back(-5.2067430); //n for valid values 1
    ub1.push_back(-28.320333); //n for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //n for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //n for valid values 2
    ub3.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 1
    ub3.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 2
    ub4.push_back(-numeric_limits<double>::infinity()); //beta for valid values 1
    ub4.push_back(-numeric_limits<double>::infinity()); //beta for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);
    ub.push_back(ub4);

    return ub;
  }

  template <class T_n, class T_N, 
      class T_size1, class T_size2, 
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  typename stan::return_type<T_size1, T_size2>::type 
  log_prob(const T_n& n, const T_N& N, 
     const T_size1& alpha, const T_size2& beta, 
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::beta_binomial_log(n, N, alpha, beta);
  }

  template <bool propto, 
      class T_n, class T_N, 
      class T_size1, class T_size2, 
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  typename stan::return_type<T_size1, T_size2>::type 
  log_prob(const T_n& n, const T_N& N, 
     const T_size1& alpha, const T_size2& beta, 
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::beta_binomial_log<propto>(n, N, alpha, beta);
  }
  
  
  template <class T_n, class T_N, 
      class T_size1, class T_size2, 
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  var log_prob_function(const T_n& n, const T_N& N, 
      const T_size1& alpha, const T_size2& beta, 
      const T4&, const T5&, const T6&, 
      const T7&, const T8&, const T9&) {
    using stan::math::lbeta;
    using stan::math::binomial_coefficient_log;
    using stan::prob::include_summand;

    var logp(0);
    if (n < 0 || n > N)
      return logp;
    
    if (include_summand<true>::value)
      logp += binomial_coefficient_log(N,n);
    if (include_summand<true,T_size1,T_size2>::value)
      logp += lbeta(n + alpha, N - n + beta) 
  - lbeta(alpha,beta);
    return logp;
  }
};

TEST(ProbDistributionsBetaBinomialCDF,Values) {
    EXPECT_FLOAT_EQ(0.8868204314, stan::prob::beta_binomial_cdf(49, 100, 1.349, 3.938));
}
