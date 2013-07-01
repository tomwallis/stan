// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/inv_gamma.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsInvGamma : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-1.0);       // expected log_prob

    param[0] = 0.5;                 // y
    param[1] = 2.9;                 // alpha
    param[2] = 3.1;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-0.8185295); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // alpha
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // beta
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
    lb.push_back(1.0e-300); //y
    lb.push_back(1.0e-300); //alpha
    lb.push_back(1.0e-300); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    lb2.push_back(-691.77556); //alpha for valid values 1
    lb2.push_back(-696.28240); //alpha for valid values 2
    lb3.push_back(-690.77556); //beta for valid values 1
    lb3.push_back(-2001.1486); //beta for valid values 2

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //alpha
    ub.push_back(numeric_limits<double>::infinity()); //beta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
   
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 2
    ub3.push_back(-numeric_limits<double>::infinity()); //beta for valid values 1
    ub3.push_back(-numeric_limits<double>::infinity()); //beta for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }

  template <typename T_y, typename T_shape, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::inv_gamma_log(y, alpha, beta);
  }

  template <bool propto, 
      typename T_y, typename T_shape, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::inv_gamma_log<propto>(y, alpha, beta);
  }
  
  
  template <typename T_y, typename T_shape, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_shape& alpha, const T_scale& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    if (y <= 0)
      return stan::prob::LOG_ZERO;
    
    using boost::math::lgamma;
    using stan::math::multiply_log;
    using stan::prob::include_summand;
    
    var lp = 0.0;
    if (include_summand<true,T_shape>::value)
      lp -= lgamma(alpha);
    if (include_summand<true,T_shape,T_scale>::value)
      lp += multiply_log(alpha,beta);
    if (include_summand<true,T_y,T_shape>::value)
      lp -= multiply_log(alpha+1.0, y);
    if (include_summand<true,T_y,T_scale>::value)
      lp -= beta / y;
    return lp;
  }
};

TEST(ProbDistributionsInvGammaCdf,Values) {
    EXPECT_FLOAT_EQ(0.557873, stan::prob::inv_gamma_cdf(4.39, 1.349, 3.938));
}
