// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/scaled_inv_chi_square.hpp>

#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsScaledInvChiSquare : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 12.7;          // y
    param[1] = 6.1;           // nu
    param[2] = 3.0;           // s
    parameters.push_back(param);
    log_prob.push_back(-3.091965); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.5;           // s
    parameters.push_back(param);
    log_prob.push_back(-1.737086); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    
    // nu
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // s
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
    lb.push_back(1.0e-300); //nu
    lb.push_back(1.0e-300); //s

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    lb2.push_back(-694.0102770722383); //nu for valid values 1
    lb2.push_back(-691.4686750787737); //nu for valid values 2
    lb3.push_back(-4221.362803286064); //s for valid values 1
    lb3.push_back(-691.6944664314184); //s for valid values 2
  
    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //nu
    ub.push_back(numeric_limits<double>::infinity()); //s

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
   
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //nu for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //nu for valid values 2
    ub3.push_back(-numeric_limits<double>::infinity()); //s for valid values 1
    ub3.push_back(-numeric_limits<double>::infinity()); //s for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }

  template <class T_y, class T_dof, class T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  log_prob(const T_y& y, const T_dof& nu, const T_scale& s,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::scaled_inv_chi_square_log(y, nu, s);
  }

  template <bool propto, 
      class T_y, class T_dof, class T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  log_prob(const T_y& y, const T_dof& nu, const T_scale& s,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::scaled_inv_chi_square_log<propto>(y, nu, s);
  }
  
  
  template <class T_y, class T_dof, class T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_dof& nu, const T_scale& s,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using stan::prob::include_summand;
    using stan::math::multiply_log;
    using stan::math::square;

    
    if (y <= 0)
      return stan::prob::LOG_ZERO;
    
    var logp(0);
    if (include_summand<true,T_dof>::value) {
      var half_nu = 0.5 * nu;
      logp += multiply_log(half_nu,half_nu) - lgamma(half_nu);
    }
    if (include_summand<true,T_dof,T_scale>::value)
      logp += nu * log(s);
    if (include_summand<true,T_dof,T_y>::value)
      logp -= multiply_log(nu*0.5+1.0, y);
    if (include_summand<true,T_dof,T_y,T_scale>::value)
      logp -= nu * 0.5 * square(s) / y;
    return logp;
  }
};

TEST(ProbDistributionsScaledInvChiSquareCDF, Values) {
    EXPECT_FLOAT_EQ(0.37242326, stan::prob::scaled_inv_chi_square_cdf(4.39, 1.349, 1.984));
}
