// Arguments: Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <stan/math/functions/log1m.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBernoulli : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.25)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.75)); // expected log_prob

    param[0] = 1;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.01)); // expected log_prob

    param[0] = 0;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);
    log_prob.push_back(log(0.99)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1);

    index.push_back(0U);
    value.push_back(2);

    // theta
    index.push_back(1U);
    value.push_back(-0.001);

    index.push_back(1U);
    value.push_back(1.001);
  }


 double num_params() {
    return 2;
  }

  std::vector<double> lower_bounds() {
    std::vector<double> lb;
    lb.push_back(0); //n
    lb.push_back(0); //theta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
   
    lb1.push_back(-0.28768207); //n for valid values 1
    lb1.push_back(-0.28768207); //n for valid values 2
    lb1.push_back(-0.010050336); //n for valid values 3
    lb1.push_back(-0.010050336); //n for valid values 4
    lb2.push_back(-numeric_limits<double>::infinity()); //theta for valid values 1
    lb2.push_back(0.0); //theta for valid values 2
    lb2.push_back(-numeric_limits<double>::infinity()); //theta for valid values 3
    lb2.push_back(0.0); //theta for valid values 4

    lb.push_back(lb1);
    lb.push_back(lb2);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(1); //n
    ub.push_back(1); //theta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
   
    ub1.push_back(-1.3862943); //n for valid values 1
    ub1.push_back(-1.3862943); //n for valid values 2
    ub1.push_back(-4.6051703); //n for valid values 3
    ub1.push_back(-4.6051703); //n for valid values 4
    ub2.push_back(0.0); //theta for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //theta for valid values 2
    ub2.push_back(0.0); //theta for valid values 3
    ub2.push_back(-numeric_limits<double>::infinity()); //theta for valid values 4

    ub.push_back(ub1);
    ub.push_back(ub2);

    return ub;
  }


  template <class T_n, class T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_n, T_prob>::type 
  log_prob(const T_n& n, const T_prob& theta, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::bernoulli_log(n, theta);
  }

  template <bool propto, 
      class T_n, class T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_n, T_prob>::type 
  log_prob(const T_n& n, const T_prob& theta, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::bernoulli_log<propto>(n, theta);
  }
  
  
  template <class T_n, class T_prob, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_n& n, const T_prob& theta, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using stan::math::log1m;
    using stan::prob::include_summand;

    var logp(0);
    if (include_summand<true,T_prob>::value) {
      if (n == 1)
  logp += log(theta);
      else if (n == 0)
  logp += log1m(theta);
    }
    return logp;
  }
};

TEST(ProbDistributionsBernoulliCDF,Values) {
    EXPECT_FLOAT_EQ(1, stan::prob::bernoulli_cdf(1, 0.57));
    EXPECT_FLOAT_EQ(1 - 0.57, stan::prob::bernoulli_cdf(0, 0.57));
}
