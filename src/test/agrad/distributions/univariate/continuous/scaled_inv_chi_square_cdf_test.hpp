// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/scaled_inv_chi_square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfScaledInvChiSquare : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 3.0;           // y
    param[1] = 0.5;           // nu (Degrees of Freedom)
    param[2] = 3.3;           // s  (Scale)
    parameters.push_back(param);
    cdf.push_back(0.0781210912);  // expected CDF

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
 
    // nu
    index.push_back(1U);
    value.push_back(-1.0);
      
    index.push_back(1U);
    value.push_back(0.0);
      
    // s
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);
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
   
    lb1.push_back(0.0); //y for valid values 1
    lb2.push_back(0.0); //nu for valid values 1
    lb3.push_back(1.0); //s for valid values 1
  
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
   
    ub1.push_back(1.0); //y for valid values 1
    ub2.push_back(0.0); //nu for valid values 1
    ub3.push_back(0.0); //s for valid values 1

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }
    
  template <typename T_y, typename T_dof, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  cdf(const T_y& y, const T_dof& nu, const T_scale& s,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::scaled_inv_chi_square_cdf(y, nu, s);
  }


  
  template <typename T_y, typename T_dof, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  cdf_function(const T_y& y, const T_dof& nu, const T_scale& s,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      return stan::prob::scaled_inv_chi_square_cdf(y, nu, s);
  }
    
};
