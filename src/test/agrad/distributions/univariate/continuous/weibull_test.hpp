// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/weibull.hpp>

#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsWeibull : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 2.0;                 // y
    param[1] = 0.5;                 // alpha
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.4539344); // expected log_prob

    param[0] = 0.25;                // y
    param[1] = 2.9;                 // alpha
    param[2] = 1.8;                 // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.277094);  // expected log_prob

    param[0] = 3.9;                 // y
    param[1] = 1.7;                 // alpha
    param[2] = 0.25;                // sigma
    parameters.push_back(param);
    log_prob.push_back(-102.8962);  // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(0U);
    value.push_back(-1.0);

    // alpha
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(0.0);

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);
  }

  double num_params() {
    return 3;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0.0); //y
    lb.push_back(1.0e-300); //alpha
    lb.push_back(1.0e-300); //sigma

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(numeric_limits<double>::infinity()); //y for valid values 1
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 3
    lb2.push_back(-692.4686751); //alpha for valid values 1
    lb2.push_back(-690.3892335); //alpha for valid values 2
    lb2.push_back(-693.1365045); //alpha for valid values 3
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 1
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 2
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 3

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //alpha
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
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 3
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 1
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 2
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 3

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
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::weibull_log(y, alpha, sigma);
  }

  template <bool propto, 
      typename T_y, typename T_shape, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  log_prob(const T_y& y, const T_shape& alpha, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::weibull_log<propto>(y, alpha, sigma);
  }
  
  
  template <typename T_y, typename T_shape, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_shape& alpha, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using std::pow;
    using stan::math::multiply_log;
    using stan::math::value_of;
    using stan::prob::include_summand;
    
    var logp(0);
    
    if (include_summand<true,T_shape>::value)
      logp += log(alpha);
    if (include_summand<true,T_y,T_shape>::value)
      logp += multiply_log(alpha-1.0, y);
    if (include_summand<true,T_shape,T_scale>::value)
      logp -= multiply_log(alpha, sigma);
    if (include_summand<true,T_y,T_shape,T_scale>::value)
      logp -= pow(y / sigma, alpha);
    return logp;
  }
};

TEST(ProbDistributionsWeibull,Cumulative) {
  using stan::prob::weibull_cdf;
  using std::numeric_limits;
  EXPECT_FLOAT_EQ(0.86466472, weibull_cdf(2.0,1.0,1.0));
  EXPECT_FLOAT_EQ(0.0032585711, weibull_cdf(0.25,2.9,1.8));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(3.9,1.7,0.25));

  // ??
  EXPECT_FLOAT_EQ(0.0,
                  weibull_cdf(-numeric_limits<double>::infinity(),
                              1.0,1.0));
  EXPECT_FLOAT_EQ(0.0, weibull_cdf(0.0,1.0,1.0));
  EXPECT_FLOAT_EQ(1.0, weibull_cdf(numeric_limits<double>::infinity(),
                                   1.0,1.0));
}
