// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/beta.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfBeta : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 0.5;           // y
    param[1] = 4.4;           // alpha (Success Scale)
    param[2] = 3.2;           // beta  (Faiulre Scale)
    parameters.push_back(param);
    cdf.push_back(0.3223064740892);  // expected CDF

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // y

    // alpha
    index.push_back(1U);
    value.push_back(-1.0);
      
    index.push_back(1U);
    value.push_back(0.0);
      
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
      
    // beta
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

  }
  
  double num_params() {
    return 3;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0.0); //y
    lb.push_back(1.0e-300); //alpha
    lb.push_back(1.0e-300); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(0.0); //y for valid values 1
    lb2.push_back(1.0); //alpha for valid values 1
    lb3.push_back(0.0); //beta for valid values 1

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(1.0); //y
    ub.push_back(numeric_limits<double>::infinity()); //alpha
    ub.push_back(numeric_limits<double>::infinity()); //beta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
   
    ub1.push_back(1.0); //y for valid values 1
    ub2.push_back(0.0); //alpha for valid values 1
    ub3.push_back(1.0); //beta for valid values 1

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }
    
  template <typename T_y, typename T_scale_succ, typename T_scale_fail,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale_succ, T_scale_fail>::type 
  cdf(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::beta_cdf(y, alpha, beta);
  }


  
  template <typename T_y, typename T_scale_succ, typename T_scale_fail,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale_succ, T_scale_fail>::type 
  cdf_function(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::beta_cdf(y, alpha, beta);
      
  }
    
};
