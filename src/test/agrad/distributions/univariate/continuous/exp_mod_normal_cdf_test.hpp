// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/exp_mod_normal.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfExpModNormal : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(4);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //lambda
    parameters.push_back(param);
    cdf.push_back(0.2384216994);     // expected cdf

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //lambda
    parameters.push_back(param);
    cdf.push_back(0.5380794103); // expected cdf

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 3; //lambda
    parameters.push_back(param);
    cdf.push_back(0.012340236); // expected cdf

    param[0] = -1.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 1.2;           // sigma
    param[3] = 1.9; //lambda
    parameters.push_back(param);
    cdf.push_back(0.00094264915); // expected cdf
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
    return 0; //FIXME: bad ub behaviors
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
   
    lb1.push_back(0.0); //y for valid values 1
    lb1.push_back(0.0); //y for valid values 2
    lb1.push_back(0.0); //y for valid values 3
    lb1.push_back(0.0); //y for valid values 4
    lb2.push_back(1.0); //mu for valid values 1
    lb2.push_back(1.0); //mu for valid values 2
    lb2.push_back(1.0); //mu for valid values 3
    lb2.push_back(1.0); //mu for valid values 4
    lb3.push_back(0.0); //sigma for valid values 1
    lb3.push_back(0.63212056); //sigma for valid values 2
    lb3.push_back(0.0); //sigma for valid values 3
    lb3.push_back(0.0); //sigma for valid values 4
    lb4.push_back(0.0); //lambda valid values 1
    lb4.push_back(0.0); //lambda valid values 2
    lb4.push_back(0.0); //lambda valid values 3
    lb4.push_back(0.0); //lambda valid values 4

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);
    lb.push_back(lb4);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(1000); //y
    ub.push_back(1000); //mu
    ub.push_back(1000); //sigma
    ub.push_back(1000); //lambda

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
    std::vector<double> ub4;
   
    ub1.push_back(1.0); //y for valid values 1
    ub1.push_back(1.0); //y for valid values 2
    ub1.push_back(1.0); //y for valid values 3
    ub1.push_back(1.0); //y for valid values 4
    ub2.push_back(1.0); //mu for valid values 1
    ub2.push_back(1.0); //mu for valid values 2
    ub2.push_back(1.0); //mu for valid values 3
    ub2.push_back(1.0); //mu for valid values 4
    ub3.push_back(1.0); //sigma for valid values 1
    ub3.push_back(1.0); //sigma for valid values 2
    ub3.push_back(1.0); //sigma for valid values 3
    ub3.push_back(1.0); //sigma for valid values 4
    ub4.push_back(1.0); //lambda valid values 1
    ub4.push_back(1.0); //lambda valid values 2
    ub4.push_back(1.0); //lambda valid values 3
    ub4.push_back(1.0); //lambda valid values 4

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
  cdf(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_inv_scale& lambda, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::exp_mod_normal_cdf(y, mu, sigma, lambda);
  }


  template <typename T_y, typename T_loc, typename T_scale,
      typename T_inv_scale, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_inv_scale>::type 
  cdf_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
         const T_inv_scale& lambda, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {

    return 0.5 * (1 + erf((y - mu) / (sqrt(2.0) * sigma))) - exp(-lambda * (y - mu) + lambda * sigma * lambda * sigma / 2.0) * (0.5 * (1 + erf(((y - mu) - sigma * lambda * sigma) / (sqrt(2.0) * sigma))));
  }
};
