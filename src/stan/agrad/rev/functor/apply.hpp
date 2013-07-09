#ifndef __STAN__AGRAD__REV__FUNCTOR__APPLY_HPP__
#define __STAN__AGRAD__REV__FUNCTOR__APPLY_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <stan/agrad/rev/op/vv_vari.hpp>
#include <stan/agrad/rev/op/vd_vari.hpp>
#include <stan/agrad/rev/op/dv_vari.hpp>

namespace stan {
  namespace agrad {
    
    namespace {

      template <class F>
      struct fun_v_vari : public op_v_vari {
        fun_v_vari(const var& x)  
        : op_v_vari(F::f(x.vi_->val_), x.vi_) {
        }
        virtual void chain() {
          avi_->adj_ += adj_ * F::dx(avi_->val_,val_);
        }
      };

      template <class F>
      struct fun_vv_vari : public op_vv_vari {
        fun_vv_vari(const var& x, const var& y)  
          : op_vv_vari(F::f(x.vi_->val_, y.vi_->val_), x.vi_, y.vi_) {  
        }
        virtual void chain() {
          avi_->adj_ += adj_ * F::dx1(avi_->val_,bvi_->val_,val_);
          bvi_->adj_ += adj_ * F::dx2(avi_->val_,bvi_->val_,val_);
        }
      };
      template <class F>
      struct fun_vd_vari : public op_vd_vari {
        fun_vd_vari(const var& x, double yv)
          : op_vd_vari(F::f(x.vi_->val_, yv), x.vi_, yv) {
        }
        virtual void chain() {
          avi_->adj_ += adj_ * F::dx1(avi_->val_,bd_,val_);
        }
      };
      template <class F>
      struct fun_dv_vari : public op_dv_vari {
        fun_dv_vari(double xv, const var& y)  
          : op_dv_vari(F::f(xv, y.vi_->val_), xv, y.vi_) {  
        }
        virtual void chain() {
          bvi_->adj_ += adj_ * F::dx2(ad_,bvi_->val_,val_);
        }
      };



    }


    /**
     * Apply functor specified by template class to specified
     * argument.
     *
     * The class <code>F</code> specified by the template parameter
     * must implement a static function with signatures:
     *
     * Function value:
     * <code>double f(double x);</code>
     *
     * Derivative at x:
     * <code>double dx(double x, double fx);</code>
     *
     * The arguments to <code>dx</code> are the values of x and of of
     * f(x), the latter of which may be used to help calculate the
     * derivative.  It is also acceptable to provide <code>const
     * double&amp;</code> types for <code>dx</code>.
     *
     * @tparam F Class defining static function f(double)
     * @param x Argument
     * @return Result of applying static function to argument,
     * F::f(x), with all derivatives calculated
     */
    template <class F>
    inline var apply(const var& x) {
      return var(new fun_v_vari<F>(x));
    }

    template <class F>
    inline var apply(const var& x1, const var& x2) {
      return var(new fun_vv_vari<F>(x1,x2));
    }
    template <class F>
    inline var apply(const var& x1, double x2) {
      return var(new fun_vd_vari<F>(x1,x2));
    }
    template <class F>
    inline var apply(double x1, const var& x2) {
      return var(new fun_dv_vari<F>(x1,x2));
    }

  }
}

#endif
