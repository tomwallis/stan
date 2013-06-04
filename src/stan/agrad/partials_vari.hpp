#ifndef __STAN__AGRAD__PARTIALS_VARI_HPP__
#define __STAN__AGRAD__PARTIALS_VARI_HPP__
#include <iostream>
#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {

    class partials_vari : public vari {
    private:
      const size_t N_;
      vari** operands_;   
      double* partials_;
    public: 
      partials_vari(double value,
                    size_t N,
                    vari** operands, double* partials)
        : vari(value),
          N_(N),
          operands_(operands),
          partials_(partials) { }
      void chain() {
        for (size_t n = 0; n < N_; ++n)
          operands_[n]->adj_ += adj_ * partials_[n];
      }
    };

    namespace {
      template<typename T>
      T partials_to_var(double logp, size_t /* nvaris */,
                        agrad::vari** /* all_varis */,
                        double* /* all_partials */) {
        return logp;
      }
      template<>
      var partials_to_var<var>(double logp, size_t nvaris,
                               agrad::vari** all_varis,
                               double* all_partials) {
        return var(new agrad::partials_vari(logp, nvaris, all_varis, all_partials));
      }

      template<typename T, 
               bool is_vec = is_vector<T>::value, 
               bool is_const = is_constant_struct<T>::value>
      struct set_varis {
        inline size_t set(agrad::vari** /*varis*/, const T& /*x*/) {
          return 0U;
        }
      };
      template<typename T>
      struct set_varis <T,true,false>{
        inline size_t set(agrad::vari** varis, const T& x) {
          for (size_t n = 0; n < length(x); n++)
            varis[n] = x[n].vi_;
          return length(x);
        }
      };
      template<>
      struct set_varis<agrad::var, false, false> {
        inline size_t set(agrad::vari** varis, const agrad::var& x) {
          varis[0] = x.vi_;
          return (1);
        }
      };
    }

    /**
     * A variable implementation that stores operands and
     * derivatives with respect to the variable.
     */
    template<typename T1=double, typename T2=double, typename T3=double, 
             typename T4=double, typename T5=double, typename T6=double, 
             typename T_return_type=typename return_type<T1,T2,T3,T4,T5,T6>::type,
             bool contains_fvar=contains_fvar<T1,T2,T3,T4,T5,T6>::value>
    struct OperandsAndPartials {
      const static bool all_constant = is_constant<T_return_type>::value;
      size_t nvaris;
      agrad::vari** all_varis;
      double* all_partials;

      VectorView<double*, is_vector<T1>::value> d_x1;
      VectorView<double*, is_vector<T2>::value> d_x2;
      VectorView<double*, is_vector<T3>::value> d_x3;
      VectorView<double*, is_vector<T4>::value> d_x4;
      VectorView<double*, is_vector<T5>::value> d_x5;
      VectorView<double*, is_vector<T6>::value> d_x6;
      
      OperandsAndPartials(const T1& x1=0, const T2& x2=0, const T3& x3=0, 
                          const T4& x4=0, const T5& x5=0, const T6& x6=0)
        : nvaris(!is_constant_struct<T1>::value * length(x1) +
                 !is_constant_struct<T2>::value * length(x2) +
                 !is_constant_struct<T3>::value * length(x3) +
                 !is_constant_struct<T4>::value * length(x4) +
                 !is_constant_struct<T5>::value * length(x5) +
                 !is_constant_struct<T6>::value * length(x6)),
          all_varis((agrad::vari**)agrad::chainable::operator new(sizeof(agrad::vari*) * nvaris)), 
          all_partials((double*)agrad::chainable::operator new(sizeof(double) * nvaris)),
          d_x1(all_partials),
          d_x2(all_partials 
               + (!is_constant_struct<T1>::value) * length(x1)),
          d_x3(all_partials 
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)),
          d_x4(all_partials 
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)),
          d_x5(all_partials 
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)
               + (!is_constant_struct<T4>::value) * length(x4)),
          d_x6(all_partials 
               + (!is_constant_struct<T1>::value) * length(x1)
               + (!is_constant_struct<T2>::value) * length(x2)
               + (!is_constant_struct<T3>::value) * length(x3)
               + (!is_constant_struct<T4>::value) * length(x4)
               + (!is_constant_struct<T5>::value) * length(x5))
      {
        size_t base = 0;
        if (!is_constant_struct<T1>::value)
          base += set_varis<T1>().set(&all_varis[base], x1);
        if (!is_constant_struct<T2>::value)
          base += set_varis<T2>().set(&all_varis[base], x2);
        if (!is_constant_struct<T3>::value)
          base += set_varis<T3>().set(&all_varis[base], x3);
        if (!is_constant_struct<T4>::value)
          base += set_varis<T4>().set(&all_varis[base], x4);
        if (!is_constant_struct<T5>::value)
          base += set_varis<T5>().set(&all_varis[base], x5);
        if (!is_constant_struct<T6>::value)
          set_varis<T6>().set(&all_varis[base], x6);
        std::fill(all_partials, all_partials+nvaris, 0);
      }

      T_return_type
      to_var(double logp) {
        return partials_to_var<T_return_type>(logp, nvaris, all_varis, all_partials);
      }
    };




    namespace {
      template <typename T, 
                bool contains_fvar = contains_fvar<T>::value>
      struct tangent {
        inline double value(VectorView<const T>& /* x */, const size_t /* index */) {
          return 0;
        }
      };

      template <typename T>
      struct tangent<T,true> {
        inline 
        typename scalar_type<T>::type::scalar_type
        value(VectorView<const T>& x, const size_t index) {
          return x[index].tangent();
        }
      };
    }

    /**
     * A variable implementation that stores operands and
     * derivatives with respect to the variable. This is a 
     * partial specialization for when it has an fvar.
     */
    template<typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, 
             typename T_return_type>
    struct OperandsAndPartials<T1,T2,T3,T4,T5,T6,
                               T_return_type,
                               true> {
      typedef typename T_return_type::scalar_type T_scalar_type;
      const static bool all_constant = is_constant<T_return_type>::value;
      size_t nx1, nx2, nx3, nx4, nx5, nx6;
      size_t npartials;
      T_scalar_type* all_partials;
      
      VectorView<T_scalar_type*, is_vector<T1>::value> d_x1;
      VectorView<T_scalar_type*, is_vector<T2>::value> d_x2;
      VectorView<T_scalar_type*, is_vector<T3>::value> d_x3;
      VectorView<T_scalar_type*, is_vector<T4>::value> d_x4;
      VectorView<T_scalar_type*, is_vector<T5>::value> d_x5;
      VectorView<T_scalar_type*, is_vector<T6>::value> d_x6;
      
      VectorView<const T1> x1_vec;
      VectorView<const T2> x2_vec;
      VectorView<const T3> x3_vec;
      VectorView<const T4> x4_vec;
      VectorView<const T5> x5_vec;
      VectorView<const T6> x6_vec;

      OperandsAndPartials(const T1& x1=0, const T2& x2=0, const T3& x3=0, 
                          const T4& x4=0, const T5& x5=0, const T6& x6=0) 
        : nx1(!is_constant_struct<T1>::value * length(x1)),
          nx2(!is_constant_struct<T2>::value * length(x2)),
          nx3(!is_constant_struct<T3>::value * length(x3)),
          nx4(!is_constant_struct<T4>::value * length(x4)),
          nx5(!is_constant_struct<T5>::value * length(x5)),
          nx6(!is_constant_struct<T6>::value * length(x6)),
          npartials(nx1 + nx2 + nx3 + nx4 + nx5 + nx6),
          all_partials((T_scalar_type*)agrad::chainable::operator new(sizeof(T_scalar_type) * npartials)),
          d_x1(all_partials),
          d_x2(all_partials + nx1),
          d_x3(all_partials + nx1 + nx2),
          d_x4(all_partials + nx1 + nx2 + nx3),
          d_x5(all_partials + nx1 + nx2 + nx3 + nx4),
          d_x6(all_partials + nx1 + nx2 + nx3 + nx4 + nx5),
          x1_vec(x1),
          x2_vec(x2),
          x3_vec(x3),
          x4_vec(x4),
          x5_vec(x5),
          x6_vec(x6) {
      }

      T_return_type
      to_var(const T_scalar_type& logp) {
        T_scalar_type derivative(0);
        for (size_t n = 0; n < nx1; n++) 
          derivative += d_x1[n] * tangent<T1>().value(x1_vec, n);
        for (size_t n = 0; n < nx2; n++) 
          derivative += d_x2[n] * tangent<T2>().value(x2_vec, n);
        for (size_t n = 0; n < nx3; n++) 
          derivative += d_x3[n] * tangent<T3>().value(x3_vec, n);
        for (size_t n = 0; n < nx4; n++) 
          derivative += d_x4[n] * tangent<T4>().value(x4_vec, n);
        for (size_t n = 0; n < nx5; n++) 
          derivative += d_x5[n] * tangent<T5>().value(x5_vec, n);
        for (size_t n = 0; n < nx6; n++) 
          derivative += d_x6[n] * tangent<T6>().value(x6_vec, n);
        

        return T_return_type(logp, derivative);
      }
    };

  } 
} 


#endif
