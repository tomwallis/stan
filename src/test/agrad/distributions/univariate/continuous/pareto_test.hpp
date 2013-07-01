// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/pareto.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsPareto : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.5;           // y
    param[1] = 0.5;           // y_min
    param[2] = 2.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-1.909543); // expected log_prob

    param[0] = 19.5;          // y
    param[1] = 0.5;          // y_min
    param[2] = 5.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-19.67878); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    // y_min
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // alpha
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

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
   
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    lb1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    lb2.push_back(-461.04026); //y_min for valid values 1
    lb2.push_back(-1167.5056); //y_min for valid values 2
    lb3.push_back(-230.66398); //alpha for valid values 1
    lb3.push_back(-233.22892); //alpha for valid values 2

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
   
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 1
    ub1.push_back(-numeric_limits<double>::infinity()); //y for valid values 2
    ub2.push_back(-numeric_limits<double>::infinity()); //y_min for valid values 1
    ub2.push_back(-numeric_limits<double>::infinity()); //y_min for valid values 2
    ub3.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 1
    ub3.push_back(-numeric_limits<double>::infinity()); //alpha for valid values 2

    ub.push_back(ub1);
    ub.push_back(ub2);
    ub.push_back(ub3);

    return ub;
  }

  template <class T_y, class T_scale, class T_shape,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_scale& y_min, const T_shape& alpha,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::pareto_log(y, y_min, alpha);
  }

  template <bool propto, 
      class T_y, class T_scale, class T_shape,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_scale& y_min, const T_shape& alpha,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::pareto_log<propto>(y, y_min, alpha);
  }
  

  template <class T_y, class T_scale, class T_shape,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, typename T9>
  var log_prob_function(const T_y& y, const T_scale& y_min, const T_shape& alpha,
      const T3&, const T4&, const T5&, 
      const T6&, const T7&, const T8&, const T9&) {
      using stan::math::multiply_log;
      using stan::prob::include_summand;
      using stan::prob::LOG_ZERO;

      var logp(0);
      if (include_summand<true,T_y,T_scale>::value)
  if (y < y_min)
    return LOG_ZERO;
      if (include_summand<true,T_shape>::value)
  logp += log(alpha);
      if (include_summand<true,T_scale,T_shape>::value)
  logp += multiply_log(alpha, y_min);
      if (include_summand<true,T_y,T_shape>::value)
  logp -= multiply_log(alpha+1.0, y);
      return logp;
  }
};

TEST(ProbDistributionsParetoCDF, Values) {
    EXPECT_FLOAT_EQ(0.60434447, stan::prob::pareto_cdf(3.45, 2.89, 5.235));
}
