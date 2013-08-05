#include <stan/prob/distributions/univariate/continuous/normal_def.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace prob {
    template double normal_log<double,double,double>
    (const double&, const double&, const double&);
    
    template stan::agrad::var normal_log<double,stan::agrad::var,double>
    (const double&, const stan::agrad::var&, const double&);
    template stan::agrad::var normal_log<stan::agrad::var,double,double>
    (const stan::agrad::var&, const double&, const double&);


    /*template double normal_log<true,double,double,double>
    (const double&, const double&, const double&);
    template double normal_log<false,double,double,double>
    (const double&, const double&, const double&);

    
    template stan::agrad::var normal_log<true,double,double,stan::agrad::var>
    (const double&, const double&, const stan::agrad::var&);
    template stan::agrad::var normal_log<false,double,double,stan::agrad::var>
    (const double&, const double&, const stan::agrad::var&);
    template stan::agrad::var normal_log<double,double,stan::agrad::var>
    (const double&, const double&, const stan::agrad::var&);*/
  }
}
