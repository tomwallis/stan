// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/pareto.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfPareto : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 0.75;          // y
    param[1] = 0.5;           // y_min (Scale)
    param[2] = 3.3;           // alpha (Shape)
    parameters.push_back(param);
    cdf.push_back(0.7376392612);  // expected CDF

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
 
    // y_min
    index.push_back(1U);
    value.push_back(-1.0);
      
    index.push_back(1U);
    value.push_back(0.0);
      
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
      
    // alpha
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
    lb.push_back(1.0e-300); //y
    lb.push_back(1.0e-100); //y_min
    lb.push_back(1.0e-100); //alpha

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(0.0); //y for valid values 1
    lb2.push_back(1.0); //y_min for valid values 1
    lb3.push_back(0.0); //alpha for valid values 1

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //y_min
    ub.push_back(numeric_limits<double>::infinity()); //alpha

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
   
    ub1.push_back(1.0); //y for valid values 1
    ub2.push_back(0.0); //y_min for valid values 1
    ub3.push_back(1.0); //alpha for valid values 1

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }
    
  template <typename T_y, typename T_scale, typename T_shape,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  cdf(const T_y& y, const T_scale& y_min, const T_shape& alpha,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::pareto_cdf(y, y_min, alpha);
  }


  
  template <typename T_y, typename T_scale, typename T_shape,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  cdf_function(const T_y& y, const T_scale& y_min, const T_shape& alpha,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      using std::exp;
      using std::log;
      return 1.0 - exp( alpha * log( y_min / y ) );
  }
    
};
