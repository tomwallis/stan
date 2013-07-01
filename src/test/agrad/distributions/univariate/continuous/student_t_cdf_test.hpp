// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/student_t.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfStudentT : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(4);

    param[0] = 5.0;           // y
    param[1] = 1.5;           // nu (Degrees of Freedom)
    param[2] = 3.3;           // mu (Location)
    param[3] = 1.0;           // sigma (Scale)
    parameters.push_back(param);
    cdf.push_back(0.86466887792);  // expected CDF
     
    param[0] = 2.5;           // y
    param[1] = 3.5;           // nu (Degrees of Freedom)
    param[2] = 3.3;           // mu (Location)
    param[3] = 1.0;           // sigma (Scale)
    parameters.push_back(param);
    cdf.push_back(0.23723278834);  // expected CDF
      
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
 
    // nu
    index.push_back(1U);
    value.push_back(-1.0);
      
    index.push_back(1U);
    value.push_back(0.0);
      
    // mu

    // sigma
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
    lb.push_back(-std::numeric_limits<double>::infinity()); //y
    lb.push_back(1.0); //FIXME: nu bad behavior at lb
    lb.push_back(-std::numeric_limits<double>::infinity()); //mu
    lb.push_back(1.0); //FIXME: sigma bad behavior at lb

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
    lb2.push_back(0.83074696); //nu for valid values 1
    lb2.push_back(0.28522329); //nu for valid values 2
    lb3.push_back(0.0); //mu for valid values 1
    lb3.push_back(0.0); //mu for valid values 2
    lb4.push_back(0.86466887); //sigma for valid values 1
    lb4.push_back(0.23723279); //sigma for valid values 2

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);
    lb.push_back(lb4);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(std::numeric_limits<double>::infinity()); //y
    ub.push_back(std::numeric_limits<double>::infinity()); //nu
    ub.push_back(std::numeric_limits<double>::infinity()); //mu
    ub.push_back(std::numeric_limits<double>::infinity()); //sigma

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
    ub2.push_back(1.0); //nu for valid values 1
    ub2.push_back(1.0); //nu for valid values 2
    ub3.push_back(0.0); //mu for valid values 1
    ub3.push_back(0.0); //mu for valid values 2
    ub4.push_back(0.0); //sigma for valid values 1
    ub4.push_back(0.0); //sigma for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);
    ub.push_back(ub4);

    return ub;
  }
    
  template <typename T_y, typename T_dof, typename T_loc, typename T_scale, 
        typename T4, typename T5, typename T6, 
        typename T7, typename T8, typename T9>
  typename stan::return_type<T_y, T_dof, T_loc, T_scale>::type
  cdf(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma,
      const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::student_t_cdf(y, nu, mu, sigma);
  }


  
  template <typename T_y, typename T_dof, typename T_loc, typename T_scale,
      typename T4, typename T5, typename T6, 
        typename T7, typename T8, typename T9>
  typename stan::return_type<T_y, T_dof, T_loc, T_scale>::type 
  cdf_function(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma,
         const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      return stan::prob::student_t_cdf(y, nu, mu, sigma);
  }
    
};
