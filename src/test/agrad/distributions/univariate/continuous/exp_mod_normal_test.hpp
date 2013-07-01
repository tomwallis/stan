// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/exp_mod_normal.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionExpModNormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 0.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // lambda
    parameters.push_back(param);
    log_prob.push_back(-1.34102164500926350577078307323252902154767190882327); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 1.0;           // lambda
    parameters.push_back(param);
    log_prob.push_back(-1.1931471805599453); // expected log_prob

    param[0] = -2.0;          // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // sigma
    param[3] = 2.0;           // lambda
    parameters.push_back(param);
    log_prob.push_back(-3.66695430596734551844348822271447899701975756228145); // expected log_prob

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // sigma
    param[3] = 2.9;           // lambda
    parameters.push_back(param);
    log_prob.push_back(-3.2116852); // expected log_prob
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

    //lambda
    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(0.0);
  }

  double num_params() {
    return 4;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(-numeric_limits<double>::infinity()); //y
    lb.push_back(-numeric_limits<double>::infinity()); //mu
    lb.push_back(1.0e-300); //sigma
    lb.push_back(1.0e-300); //lambda


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
    lb3.push_back(-0.6931471); //sigma for valid values 1
    lb3.push_back(-1.0); //sigma for valid values 2
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 3
    lb3.push_back(-numeric_limits<double>::infinity()); //sigma for valid values 4
    lb4.push_back(-691.46866); //lambda valid values 1
    lb4.push_back(-690.94823); //lambda valid values 2
    lb4.push_back(-694.55866); //lambda valid values 3
    lb4.push_back(-692.260); //lambda valid values 4

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
    ub.push_back(numeric_limits<double>::infinity()); //lambda

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
    ub4.push_back(-numeric_limits<double>::infinity()); //lambda valid values 1
    ub4.push_back(-numeric_limits<double>::infinity()); //lambda valid values 2
    ub4.push_back(-numeric_limits<double>::infinity()); //lambda valid values 3
    ub4.push_back(-numeric_limits<double>::infinity()); //lambda valid values 4

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);
    ub.push_back(ub4);

    return ub;
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T_inv_scale, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_inv_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_inv_scale& lambda, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::exp_mod_normal_log(y, mu, sigma, lambda);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T_inv_scale, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale, T_inv_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T_inv_scale& lambda, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::exp_mod_normal_log<propto>(y, mu, sigma, lambda);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T_inv_scale, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_inv_scale& lambda, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;

    var lp(0.0);

    if (include_summand<true>::value)
      lp -= log(2);
    if (include_summand<true, T_inv_scale>::value)
      lp += log(lambda);
    if (include_summand<true,T_y,T_loc,T_scale,T_inv_scale>::value)
      lp += lambda * (mu + 0.5 * lambda * sigma * sigma - y) + log(erfc((mu + lambda * sigma * sigma - y) / (sqrt(2.0) * sigma)));
    return lp;
  }
};

