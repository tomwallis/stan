// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/logistic.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfLogistic : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 3.0;           // y
    param[1] = 0.5;           // Location
    param[2] = 3.3;           // Scale
    parameters.push_back(param);
    cdf.push_back(0.680827173);  // expected cdf

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
 
    // mu
      
    // sigma
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
    lb.push_back(-numeric_limits<double>::infinity()); //y
    lb.push_back(-numeric_limits<double>::infinity()); //mu
    lb.push_back(1.0e-100); //sigma

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(0.0); //y for valid values 1
    lb2.push_back(1.0); //mu for valid values 1
    lb3.push_back(1.0); //sigma for valid values 1

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
   
    ub1.push_back(1.0); //y for valid values 1
    ub2.push_back(0.0); //mu for valid values 1
    ub3.push_back(0.5); //sigma for valid values 1

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
  cdf(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::logistic_cdf(y, mu, sigma);
  }


  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  cdf_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      return 1.0 / (1.0 + exp( - (y - mu) / sigma ) );
  }
    
};
