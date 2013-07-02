// Arguments: Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfPoisson : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = 13.0;         // lambda
    parameters.push_back(param);
    cdf.push_back(0.890465); // expected cdf

    param[0] = 82;           // n
    param[1] = 42.0;         // lambda
    parameters.push_back(param);
    cdf.push_back(0.99999998); // expected cdf
    
    param[0] = 0.0;          // n
    param[1] = 3.0;          // lambda
    parameters.push_back(param);
    cdf.push_back(0.04978707); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // lambda
    index.push_back(1U);
    value.push_back(-1e-5);

    index.push_back(1U);
    value.push_back(-1);
  }
  
 double num_params() {
    return 2;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0); //n
    lb.push_back(0.0); //lambda

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
   
    lb1.push_back(0.0); //n for valid values 1
    lb1.push_back(0.0); //n for valid values 2
    lb1.push_back(0.049787068); //n for valid values 3
    lb2.push_back(1.0); //lambda for valid values 1
    lb2.push_back(1.0); //lambda for valid values 2
    lb2.push_back(1.0); //lambda for valid values 3

    lb.push_back(lb1);
    lb.push_back(lb2);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(1000); //n
    ub.push_back(numeric_limits<double>::infinity()); //lambda

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
   
    ub1.push_back(1.0); //n for valid values 1
    ub1.push_back(1.0); //n for valid values 2
    ub1.push_back(1.0); //n for valid values 3
    ub2.push_back(0.0); //lambda for valid values 1
    ub2.push_back(0.0); //lambda for valid values 2
    ub2.push_back(0.0); //lambda for valid values 3

    ub.push_back(ub1);
    ub.push_back(ub2);

    return ub;
  }

  template <typename T_n, typename T_rate, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_rate>::type
  cdf(const T_n& n, const T_rate& lambda, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::poisson_cdf(n, lambda);
  }


  template <typename T_n, typename T_rate, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_rate>::type
  cdf_function(const T_n& n, const T_rate& lambda, const T2&,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::pow;
    using stan::agrad::pow;
    using stan::agrad::lgamma;
    using boost::math::lgamma;
    using std::exp;
    using stan::agrad::exp;
    using std::log;
    
    typename stan::return_type<T_rate>::type cdf(0);
    for (int i = 0; i <= n; i++) {
      cdf += exp(i * log(lambda) - lgamma(i+1));
    }
    cdf *= exp(-lambda);
    return cdf;
  }
};
