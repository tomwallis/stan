// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/double_exponential.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsDoubleExponential : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);
    
    param[0] = 1.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.6931472);  // expected log_prob

    param[0] = 2.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.693147);   // expected log_prob
    
    param[0] = -3.0;                 // y
    param[1] = 2.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-5.693147);   // expected log_prob
    
    param[0] = 1.0;                  // y
    param[1] = 0.0;                  // mu
    param[2] = 2.0;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.886294);   // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.5;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.8);        // expected log_prob

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.25;                  // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.9068528);        // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

 double num_params() {
    return 3;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(-numeric_limits<double>::infinity()); //y
    lb.push_back(-numeric_limits<double>::infinity()); //mu
    lb.push_back(1.0e-300); //sigma

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 3
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 4
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 5
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 6
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 1
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 2
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 3
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 4
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 5
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 6
    lb3.push_back(690.08238071765375990); //sigma for valid values 1
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 2
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 3
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 4
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 5
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 6

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //mu
    ub.push_back(numeric_limits<double>::infinity()); //sigma

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
   
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 3
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 4
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 5
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 6
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 3
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 4
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 5
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 6
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 1
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 2
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 3
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 4
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 5
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 6

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::double_exponential_log(y, mu, sigma);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::double_exponential_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using std::fabs;
    using stan::prob::include_summand;
    using stan::prob::NEG_LOG_TWO;

    var logp(0);
    
    if (include_summand<true>::value)
      logp += NEG_LOG_TWO;
    if (include_summand<true,T_scale>::value)
      logp -= log(sigma);
    if (include_summand<true,T_y,T_loc,T_scale>::value)
      logp -= fabs(y - mu) / sigma;
    return logp;
  }
};


TEST(ProbDistributionsDoubleExponential,Cumulative) {
  EXPECT_FLOAT_EQ(0.5, stan::prob::double_exponential_cdf(1.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.8160603, stan::prob::double_exponential_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.003368973, stan::prob::double_exponential_cdf(-3.0,2.0,1.0));
  EXPECT_FLOAT_EQ(0.6967347, stan::prob::double_exponential_cdf(1.0,0.0,2.0));
  EXPECT_FLOAT_EQ(0.2246645, stan::prob::double_exponential_cdf(1.9,2.3,0.5));
  EXPECT_FLOAT_EQ(0.10094826, stan::prob::double_exponential_cdf(1.9,2.3,0.25));
}

