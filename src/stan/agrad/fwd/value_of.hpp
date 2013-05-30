#ifndef __STAN__AGRAD__FWD__VALUE_OF_HPP__
#define __STAN__AGRAD__FWD__VALUE_OF_HPP__

#include <stan/math/functions/value_of.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {
  namespace agrad {

    template <class T>
    inline double value_of(const fvar<T>& v) {
      using stan::math::value_of;
      return value_of(v.val_);
    }

  }
}
#endif
