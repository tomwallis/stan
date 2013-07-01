// Arguments: Ints, Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>
#include <boost/math/special_functions/binomial.hpp>

#include <stan/math/functions/lbeta.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfBetaBinomial : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(4);

    param[0] = 17;         // n
    param[1] = 45;         // N
    param[2] = 13;         // alpha
    param[3] = 15;         // beta
    parameters.push_back(param);
    cdf.push_back(0.26805232961); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // N
    index.push_back(1U);
    value.push_back(-1);
      
    // alpha
    index.push_back(2U);
    value.push_back(-1);
      
    // beta
    index.push_back(3U);
    value.push_back(-1);

  }
  
  double num_params() {
    return 4;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0.0); //n
    lb.push_back(0.0); //N
    lb.push_back(1.0e-300); //alpha
    lb.push_back(1.0e-300); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
    std::vector<double> lb4;
   
    lb1.push_back(0.0); //n for valid values 1
    lb2.push_back(1.0); //N for valid values 1
    lb3.push_back(1.0); //alpha for valid values 1
    lb4.push_back(1.0); //beta for valid values 1

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);
    lb.push_back(lb4);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<int>::infinity()); //n
    ub.push_back(numeric_limits<int>::infinity()); //N
    ub.push_back(numeric_limits<double>::infinity()); //alpha
    ub.push_back(numeric_limits<double>::infinity()); //beta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
    std::vector<double> ub4;
   
    ub1.push_back(0.0); //n for valid values 1
    ub2.push_back(1.0); //N for valid values 1
    ub3.push_back(0.0); //alpha for valid values 1
    ub4.push_back(1.0); //beta for valid values 1

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);
    ub.push_back(ub4);

    return ub;
  }

  template <typename T_n, typename T_N, typename T_size1, typename T_size2, 
        typename T4, typename T5, typename T6, 
        typename T7, typename T8, typename T9>
  typename stan::return_type<T_size1,T_size2>::type
  cdf(const T_n& n, const T_N& N, const T_size1& alpha, const T_size2& beta, 
      const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::beta_binomial_cdf(n, N, alpha, beta);
  }


  template <typename T_n, typename T_N, typename T_size1, typename T_size2, 
        typename T4, typename T5, typename T6, 
        typename T7, typename T8, typename T9>
  typename stan::return_type<T_size1,T_size2>::type
  cdf_function(const T_n& n, const T_N& N, const T_size1& alpha, const T_size2& beta, 
               const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {

    using std::exp;
    using stan::math::lbeta;
    using boost::math::binomial_coefficient;
      
    typename stan::return_type<T_size1,T_size2>::type cdf(0);
 
    for (int i = 0; i <= n; i++) {
      cdf += binomial_coefficient<double>(N, i) 
             * exp( lbeta(alpha + i, N - i + beta) - lbeta(alpha, beta) );
    }
      
    return cdf;
      
  }
};
