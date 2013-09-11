#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <stan/agrad/var.hpp>

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

struct push_back_state_and_time
{
  std::vector< state_type >& m_states;
  std::vector< stan::agrad::var >& m_times;

  push_back_state_and_time( std::vector< state_type > &states , std::vector< stan::agrad::var > &times )
    : m_states( states ) , m_times( times ) { }

  void operator()( const state_type &x , stan::agrad::var t )
  {
    m_states.push_back( x );
    m_times.push_back( t );
  }
  
  void print() {
    for (size_t n = 0; n < m_states.size(); n++) {
      std::cout << m_times[n] << ": (" 
                << m_states[n][0] << ", " << m_states[n][1] << ")" << std::endl;
    }
  }
};

namespace stan {
  namespace agrad {
    stan::agrad::var max(stan::agrad::var a, stan::agrad::var b) {
      //return a > b ? a : b;
      return fmax(a, b);
    }
  }
}

TEST(ode,harmonic_oscillator) {
  using namespace std;
  using namespace boost::numeric::odeint;

  state_type x(2);
  x[0] = 1.0; // start at x=1.0, p=0.0
  x[1] = 0.0;

  vector<state_type> x_vec;
  vector<stan::agrad::var> times;

  stan::agrad::var gam = 0.15;
  harm_osc ho(gam);

  stan::agrad::var times_start = 0.0;
  stan::agrad::var times_end = 10.0;
  stan::agrad::var dt = 0.1;
  push_back_state_and_time obs(x_vec, times);

  size_t steps = integrate_const(make_dense_output(1.0e-6, 1.0e-6, 
                                                   runge_kutta_dopri5< state_type,
                                                   stan::agrad::var,
                                                   state_type,
                                                   stan::agrad::var>() ) , 
                                 ho, x, times_start, times_end, dt,
                                 obs);
  obs.print();
  
  std::cout << "steps: " << steps << std::endl;
}

