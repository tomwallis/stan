// Arguments: Ints, Ints, Ints, Ints
#include <stan/prob/distributions/univariate/discrete/hypergeometric.hpp>

#include <stan/math/functions/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsNegBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 5;           // n
    param[1] = 15;          // N
    param[2] = 10;          // a
    param[3] = 10;          // b
    parameters.push_back(param);
    log_prob.push_back(-4.119424); // expected log_prob

    param[0] = 5;           // n
    param[1] = 15;          // N
    param[2] = 10;          // a
    param[3] = 10;          // b
    parameters.push_back(param);
    log_prob.push_back(-4.119424); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    // N
    index.push_back(1U);
    value.push_back(-1);

    // a
    index.push_back(2U);
    value.push_back(-1);

    // b
    index.push_back(3U);
    value.push_back(-1);
  }

  double num_params() {
    return 4; //HARD TO TEST BOUNDS
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(5.0); //n
    lb.push_back(6.0); //N
    lb.push_back(10.0); //alpha
    lb.push_back(10.0); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
    std::vector<double> lb4;
   
    lb1.push_back(-4.1194243); //n for valid values 1
    lb1.push_back(-4.1194243); //n for valid values 2
    lb2.push_back(-2.7331298); //N for valid values 1
    lb2.push_back(-2.7331298); //N for valid values 2
    lb3.push_back(-4.1194243); //alpha for valid values 1
    lb3.push_back(-4.1194243); //alpha for valid values 2
    lb4.push_back(-4.1194243); //beta for valid values 1
    lb4.push_back(-4.1194243); //beta for valid values 2

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);
    lb.push_back(lb4);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(10); //n
    ub.push_back(10); //N
    ub.push_back(10); //alpha
    ub.push_back(10); //beta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
    std::vector<double> ub4;
   
    ub1.push_back(-4.1194243); //n for valid values 1
    ub1.push_back(-4.1194243); //n for valid values 2
    ub2.push_back(-1.0679332); //n for valid values 1
    ub2.push_back(-1.0679332); //n for valid values 2
    ub3.push_back(-4.1194243); //alpha for valid values 1
    ub3.push_back(-4.1194243); //alpha for valid values 2
    ub4.push_back(-4.1194243); //beta for valid values 1
    ub4.push_back(-4.1194243); //beta for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);
    ub.push_back(ub4);

    return ub;
  }

  template <class T_n, class T_N, class T_a, class T_b,
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  typename stan::return_type<T_n,T_N,T_a,T_b>::type 
  log_prob(const T_n& n, const T_N& N, const T_a& a, const T_b& b,
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::hypergeometric_log(n, N, a, b);
  }

  template <bool propto, 
      class T_n, class T_N, class T_a, class T_b,
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  double
  log_prob(const T_n& n, const T_N& N, const T_a& a, const T_b& b,
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::hypergeometric_log<propto>(n, N, a, b);
  }
  

  template <class T_n, class T_N, class T_a, class T_b,
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  var log_prob_function(const T_n& n, const T_N& N, const T_a& a, const T_b& b,
      const T4&, const T5&, const T6&, 
      const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::binomial_coefficient_log;
    
    var logp(0);
    logp += binomial_coefficient_log(a, n)
      + binomial_coefficient_log(b, N-n)
      - binomial_coefficient_log(a+b, N);
    return logp;
  }
};
