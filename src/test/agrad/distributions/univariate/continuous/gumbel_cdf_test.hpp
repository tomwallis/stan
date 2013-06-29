// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/gumbel.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfGumbel : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 0.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    cdf.push_back(0.3678794411714423215955237701614608674458111310317678);     // expected cdf

    param[0] = 1.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    cdf.push_back(0.6922006275553463538654219971827897614906780292975447); // expected cdf

    param[0] = -2.0;          // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    cdf.push_back(0.0006179789893310934986195216040530260548886143651007); // expected cdf

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // beta
    parameters.push_back(param);
    cdf.push_back(0.1203922620798295861862650786832089422663975274508450); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu

    // beta
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
    lb.push_back(-numeric_limits<double>::infinity()); //y
    lb.push_back(-numeric_limits<double>::infinity()); //mu
    lb.push_back(1.0e-100); //beta

    return lb;
  }

  std::vector<std::vector<double> > lower_bound_vals() {
    std::vector<std::vector<double> > lb;
    std::vector<double> lb1;
    std::vector<double> lb2;
    std::vector<double> lb3;
   
    lb1.push_back(0.0); //y for valid values 1
    lb1.push_back(0.0); //y for valid values 2
    lb1.push_back(0.0); //y for valid values 3
    lb1.push_back(0.0); //y for valid values 4
    lb2.push_back(1.0); //mu for valid values 1
    lb2.push_back(1.0); //mu for valid values 2
    lb2.push_back(1.0); //mu for valid values 3
    lb2.push_back(1.0); //mu for valid values 4
    lb3.push_back(0.36787944); //beta for valid values 1
    lb3.push_back(1.0); //beta for valid values 2
    lb3.push_back(0.0); //beta for valid values 3
    lb3.push_back(0.0); //beta for valid values 4

    lb.push_back(lb1);
    lb.push_back(lb2);
    lb.push_back(lb3);

    return lb;
  }

  std::vector<double> upper_bounds() {
    std::vector<double> ub;
    ub.push_back(numeric_limits<double>::infinity()); //y
    ub.push_back(numeric_limits<double>::infinity()); //mu
    ub.push_back(numeric_limits<double>::infinity()); //beta

    return ub;
  }

  std::vector<std::vector<double> > upper_bound_vals() {
    std::vector<std::vector<double> > ub;
    std::vector<double> ub1;
    std::vector<double> ub2;
    std::vector<double> ub3;
   
    ub1.push_back(1.0); //y for valid values 1
    ub1.push_back(1.0); //y for valid values 2
    ub1.push_back(1.0); //y for valid values 3
    ub1.push_back(1.0); //y for valid values 4
    ub2.push_back(0.0); //mu for valid values 1
    ub2.push_back(0.0); //mu for valid values 2
    ub2.push_back(0.0); //mu for valid values 3
    ub2.push_back(0.0); //mu for valid values 4
    ub3.push_back(0.36787944); //beta for valid values 1
    ub3.push_back(0.36787944); //beta for valid values 2
    ub3.push_back(0.36787944); //beta for valid values 3
    ub3.push_back(0.36787944); //beta for valid values 4

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
  cdf(const T_y& y, const T_loc& mu, const T_scale& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::gumbel_cdf(y, mu, beta);
  }


  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  cdf_function(const T_y& y, const T_loc& mu, const T_scale& beta,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return exp(-exp(-(y - mu) / beta));
  }
};
