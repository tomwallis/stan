#ifndef __STAN__DIFF__REV__LOG_SUM_EXP_HPP__
#define __STAN__DIFF__REV__LOG_SUM_EXP_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/calculate_chain.hpp>
#include <stan/diff/rev/scalar/op/vv_vari.hpp>
#include <stan/diff/rev/scalar/op/vd_vari.hpp>
#include <stan/diff/rev/scalar/op/dv_vari.hpp>
#include <stan/diff/rev/scalar/operator_greater_than.hpp>
#include <stan/diff/rev/scalar/operator_not_equal.hpp>
#include <stan/math/scalar/log_sum_exp.hpp>

namespace stan {
  namespace diff {

    namespace {
      class log_sum_exp_vv_vari : public op_vv_vari {
      public:
        log_sum_exp_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::log_sum_exp(avi->val_, bvi->val_),
                     avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
          bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_);
        }
      };
      class log_sum_exp_vd_vari : public op_vd_vari {
      public:
        log_sum_exp_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::log_sum_exp(avi->val_, b),
                     avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
        }
      };
      class log_sum_exp_dv_vari : public op_dv_vari {
      public:
        log_sum_exp_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::log_sum_exp(a, bvi->val_),
                     a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * calculate_chain(bvi_->val_, val_);
        }
      };

    }

    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const stan::diff::var& a,
                           const stan::diff::var& b) {
      return var(new log_sum_exp_vv_vari(a.vi_, b.vi_));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const stan::diff::var& a,
                           const double& b) {
      return var(new log_sum_exp_vd_vari(a.vi_, b));
    }
    /**
     * Returns the log sum of exponentials.
     */
    inline var log_sum_exp(const double& a,
                           const stan::diff::var& b) {
      return var(new log_sum_exp_dv_vari(a, b.vi_));
    }

  }
}
#endif
