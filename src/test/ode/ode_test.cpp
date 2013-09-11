#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>


typedef std::vector< double > state_type;

const double gam = 0.15;


class harm_osc {

    double m_gam;

public:
    harm_osc( double gam ) : m_gam(gam) { }

    void operator() ( const state_type &x , state_type &dxdt , const double /* t */ )
    {
        dxdt[0] = x[1];
        dxdt[1] = -x[0] - m_gam*x[1];
    }
};

struct push_back_state_and_time
{
  std::vector< state_type >& m_states;
  std::vector< double >& m_times;

  push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

  void operator()( const state_type &x , double t )
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

TEST(ode,harmonic_oscillator) {
  using namespace std;
  using namespace boost::numeric::odeint;

  state_type x(2);
  x[0] = 1.0; // start at x=1.0, p=0.0
  x[1] = 0.0;

  vector<state_type> x_vec;
  vector<double> times;

  harm_osc ho(0.15);

  double times_start = 0.0;
  double times_end = 10.0;
  double dt = 0.1;
  push_back_state_and_time obs(x_vec, times);
  size_t steps = integrate_const(make_dense_output(1.0e-6, 1.0e-6, 
                                                   runge_kutta_dopri5< state_type >() ) , 
                                 ho, x, times_start, times_end, dt,
                                 obs);
  obs.print();
  
  std::cout << "steps: " << steps << std::endl;
}

