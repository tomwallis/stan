// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/inv_chi_square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfInvChiSquare : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(2);

    param[0] = 3.0;           // y
    param[1] = 0.5;           // Degrees of freedom
    parameters.push_back(param);
    cdf.push_back(0.317528);  // expected cdf

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
  }
  
 double num_params() {
    return 2;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0.0); //y
    lb.push_back(1.0e-300); //nu

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
   
    lb1.push_back(0.0); //y for valid values 1
    lb2.push_back(0.0); //nu for valid values 1

    lb.push_back(lb1);
    lb.push_back(lb2);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //nu

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
   
    ub1.push_back(1.0); //y for valid values 1
    ub2.push_back(1.0); //nu for valid values 1

    ub.push_back(ub1);
    ub.push_back(ub2);

    return ub;
  }

  template <typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof>::type 
  cdf(const T_y& y, const T_dof& nu, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::inv_chi_square_cdf(y, nu);
  }


  
  template <typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof>::type 
  cdf_function(const T_y& y, const T_dof& nu, const T2&,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      return stan::prob::inv_chi_square_cdf(y, nu);
  }
    
};
