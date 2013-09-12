#ifndef __STAN__MATH__HO_HPP__
#define __STAN__MATH__HO_HPP__

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/var.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace agrad {
    stan::agrad::var max(stan::agrad::var a, stan::agrad::var b) {
      //return a > b ? a : b;
      return fmax(a, b);
    }
  }
}


namespace stan {
  namespace math {
    
    typedef std::vector< stan::agrad::var > state_type;
    
    class harm_osc {
      stan::agrad::var m_gam;
  
    public:
      harm_osc( stan::agrad::var gam ) : m_gam(gam) { }

      void operator() ( const state_type &x , state_type &dxdt , const stan::agrad::var /* t */ )
      {
        dxdt[0] = x[1];
        dxdt[1] = -x[0] - m_gam*x[1];
      }
    };

    struct push_back_state_and_time {
      std::vector< state_type >& m_states;
      std::vector< stan::agrad::var >& m_times;
      
      push_back_state_and_time( std::vector< state_type > &states , std::vector< stan::agrad::var > &times )
        : m_states( states ) , m_times( times ) { }
      
      void operator()( const state_type &x , stan::agrad::var t ) {
        m_states.push_back( x );
        m_times.push_back( t );
      }
      
      Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> 
      get() {
        size_t N = m_states.size();
        size_t M = m_states[0].size();
        Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> xhat(N-1, M);
        
        for (size_t n = 1; n < N; n++) 
          for (size_t m = 0; m < M; m++)
            xhat(n-1,m) = m_states[n][m];
        
        return xhat;
      }
      
      void print() {
        std::cout << "time,x_0,x_1" << std::endl;
        for (size_t n = 0; n < m_states.size(); n++) {
          std::cout << m_times[n].val()
                    << "," << m_states[n][0].val()
                    << "," << m_states[n][1].val()
                    << std::endl;
        }
      }
    };
    


    Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> 
    ho(std::vector<double>& t,
       Eigen::Matrix<double,Eigen::Dynamic,1>& x0,
       stan::agrad::var& gamma) {
      using namespace std;
      using namespace boost::numeric::odeint;
  
      state_type x0_state(2);
      for (size_t n = 0; n < 2; n++)
        x0_state[n] = x0[n];
      
      
      stan::agrad::var times_start = 0.0;
      stan::agrad::var times_end = t[t.size()-1];
      stan::agrad::var dt = t[0];

      vector<state_type> x_vec;
      vector<stan::agrad::var> times;
      push_back_state_and_time obs(x_vec, times);
      
      harm_osc harm_osc(gamma);


      integrate_const(make_dense_output(1.0e-6, 1.0e-6, 
                                        runge_kutta_dopri5< state_type,
                                                            stan::agrad::var,
                                                            state_type,
                                                            stan::agrad::var>() ) , 
                      harm_osc, x0_state, times_start, times_end, dt,
                      obs);
      //std::cout << "here" << std::endl;
      //obs.print();
      return obs.get();
    }
    
  }
}

#endif
