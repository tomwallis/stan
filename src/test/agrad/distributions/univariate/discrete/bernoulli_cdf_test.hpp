// Arguments: Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfBernoulli: public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(2);

    param[0] = 0;           // Successes (out of single trial)
    param[1] = 0.75;        // Probability
    parameters.push_back(param);
    cdf.push_back(1 - param[1]); // expected cdf
      
    param[0] = 1;           // Successes (out of single trial)
    param[1] = 0.75;        // Probability
    parameters.push_back(param);
    cdf.push_back(1);       // expected cdf
      
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
      
    // p (Probability)
    index.push_back(1U);
    value.push_back(-1e-4);

    index.push_back(1U);
    value.push_back(1+1e-4);
  }
  
  double num_params() {
    return 2;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0); //n
    lb.push_back(0.0); //theta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
   
    lb1.push_back(0.25); //n for valid values 1
    lb1.push_back(0.25); //n for valid values 2
    lb2.push_back(1.0); //theta for valid values 1
    lb2.push_back(1.0); //theta for valid values 2

    lb.push_back(lb1);
    lb.push_back(lb2);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(1); //n
    ub.push_back(1.0); //theta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
   
    ub1.push_back(1.0); //n for valid values 1
    ub1.push_back(1.0); //n for valid values 2
    ub2.push_back(0.0); //theta for valid values 1
    ub2.push_back(1.0); //theta for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);

    return ub;
  }

  template <typename T_n, typename T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_prob>::type
  cdf(const T_n& n, const T_prob& theta, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::bernoulli_cdf(n, theta);
  }


  template <typename T_n, typename T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_prob>::type
  cdf_function(const T_n& n, const T_prob& theta,const T2&,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {

    if(n < 0) return 0;
    if(n < 1) return 1 - theta;
    else      return 1;
      
  }
};
