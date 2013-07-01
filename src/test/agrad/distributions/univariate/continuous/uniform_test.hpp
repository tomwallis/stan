// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsUniform : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0.1;                // y
    param[1] = -0.1;               // alpha
    param[2] = 0.8;                // beta
    parameters.push_back(param);
    log_prob.push_back(log(1/0.9));   // expected log_prob

    param[0] = 0.2;                // y
    param[1] = -0.25;              // alpha
    param[2] = 0.25;               // beta
    parameters.push_back(param);
    log_prob.push_back(log(2.0));   // expected log_prob

    param[0] = 0.05;               // y
    param[1] = -5;                 // alpha
    param[2] = 5;                  // beta
    parameters.push_back(param);
    log_prob.push_back(log(0.1));   // expected log_prob
  }
 
  void invalid_values(vector<size_t>& /*index*/, 
                      vector<double>& /*value*/) {
    // y
    
    // alpha

    // beta
  }

  double num_params() {
    return 0;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(-numeric_limits<double>::infinity()); //y
    lb.push_back(-numeric_limits<double>::infinity()); //alpha
    lb.push_back(-numeric_limits<double>::infinity()); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    lb2.push_back(0.0); //alpha for valid values 1
    lb2.push_back(0.0); //alpha for valid values 2
    lb3.push_back(0.0); //beta for valid values 1
    lb3.push_back(0.0); //beta for valid values 2

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
    ub2.push_back(0.0); //alpha for valid values 1
    ub2.push_back(0.0); //alpha for valid values 2
    ub3.push_back(0.0); //beta for valid values 1
    ub3.push_back(0.0); //beta for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }

  template <class T_y, class T_low, class T_high,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_low, T_high>::type 
  log_prob(const T_y& y, const T_low& alpha, const T_high& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::uniform_log(y, alpha, beta);
  }

  template <bool propto, 
      class T_y, class T_low, class T_high,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_low, T_high>::type 
  log_prob(const T_y& y, const T_low& alpha, const T_high& beta,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::uniform_log<propto>(y, alpha, beta);
  }
  
  
  template <class T_y, class T_low, class T_high,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_low& alpha, const T_high& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      using stan::prob::include_summand;
      using stan::prob::LOG_ZERO;

      if (y < alpha || y > beta)
        return LOG_ZERO;

      var lp(0.0);
      if (include_summand<true,T_low,T_high>::value)
          lp -= log(beta - alpha);
      return lp;

  }
};
