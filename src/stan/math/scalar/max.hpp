#ifndef __STAN__MATH__SCALAR__MAX_HPP__
#define __STAN__MATH__SCALAR__MAX_HPP__

namespace stan {
  namespace math {

    inline double max(const double a, const double b) { 
      return a > b ? a : b; 
    }

  }
}

#endif
