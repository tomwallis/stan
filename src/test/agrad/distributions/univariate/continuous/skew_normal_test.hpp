// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/skew_normal.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionSkewNormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 0.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-0.91893852); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-0.898545); // expected log_prob

    param[0] = -2.0;          // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 2.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-12.585893); // expected log_prob

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // sigma
    param[3] = 2.9;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-6.6932335); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu

    // sigma
    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    //alpha
  }

  double num_params() {
    return 4;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(-numeric_limits<double>::infinity()); //y
    lb.push_back(-numeric_limits<double>::infinity()); //mu
    lb.push_back(1.0e-300); //sigma
    lb.push_back(-numeric_limits<double>::infinity()); //alpha

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
    std::vector<double> lb4;
   
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 3
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 4
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 1
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 2
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 3
    lb2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 4
    lb3.push_back(689.8566); //sigma for valid values 1
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 2
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 3
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 4
    lb4.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 1
    lb4.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 2
    lb4.push_back(-2.2257913); //alpha for valid values 3
    lb4.push_back(-2.4811223); //alpha for valid values 4
    
    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);
    lb.push_back(lb4);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //mu
    ub.push_back(numeric_limits<double>::infinity()); //sigma
    ub.push_back(numeric_limits<double>::infinity()); //alpha

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
    std::vector<double> ub4;
   
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 3
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 4
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 3
    ub2.push_back(-numeric_limits<double>::infinity()); //mu for valid values 4
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 1
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 2
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 3
    ub3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 4
    ub4.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 1
    ub4.push_back(-0.72579136); //alpha for valid values 2
    ub4.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 3
    ub4.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 4

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);
    ub.push_back(ub4);

    return ub;
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_shape>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_shape& alpha, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::skew_normal_log(y, mu, sigma, alpha);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_shape& alpha, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::skew_normal_log<propto>(y, mu, sigma, alpha);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_shape& alpha, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;

    var logp(0.0);

    if (include_summand<true>::value)
      logp -=  0.5 * log(2.0 * boost::math::constants::pi<double>());
    if (include_summand<true, T_scale>::value)
      logp -= log(sigma);
    if (include_summand<true,T_y, T_loc, T_scale>::value)
      logp -= (y - mu) / sigma * (y - mu) / sigma * 0.5;
    if (include_summand<true,T_y,T_loc,T_scale,T_shape>::value)
      logp += log(erfc(-alpha * (y - mu) / (sigma * std::sqrt(2.0))));
    return logp;
  }
};

