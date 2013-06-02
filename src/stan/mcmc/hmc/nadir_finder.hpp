#ifndef __STAN__MCMC__NADIR__FINDER__BETA__
#define __STAN__MCMC__NADIR__FINDER__BETA__

#include <math.h>
#include <stdexcept>

#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_01.hpp>

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <class M, class BaseRNG>
    class nadir_finder: public base_mcmc {
      
    public:
      
      nadir_finder(M &m, BaseRNG& rng, std::ostream* o = 0, std::ostream* e = 0):
      base_mcmc(o, e),
      _z(m.num_params_r(), m.num_params_i()),
      _integrator(this->_out_stream),
      _hamiltonian(m, this->_err_stream),
      _rand_int(rng),
      _rand_uniform(_rand_int),
      _epsilon(1.0),
      _n_threshold_sigma(5),
      _p_scale(1.0),
      _max_iterations(1e3)
      {};
      
      void seed(const std::vector<double>& q, const std::vector<int>& r) {
        _z.q = q;
        _z.r = r;
      }
      
      void init_stepsize() {
        
        this->_hamiltonian.init(this->_z);
        this->_epsilon = 2.0 / this->_z.g.norm();
        
        ps_point z_init(this->_z);
        
        int previous_direction = 0;
        double delta = 1;
        
        while (1) {
          
          this->_z.copy_base(z_init);
          
          this->_hamiltonian.sample_p(this->_z, this->_rand_int);
          this->_hamiltonian.init(this->_z);
          
          double H0 = this->_hamiltonian.H(this->_z);
          
          this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
          
          double h = this->_hamiltonian.H(this->_z);
          if (h != h) h = std::numeric_limits<double>::infinity();
          
          double delta_H = fabs((H0 - h) / H0);
          
          if      (delta_H < 5e-4) {
            
            if (previous_direction == -1) delta /= 1.5;
            this->_epsilon = (1 + delta) * this->_epsilon;
            previous_direction = 1;
            
          } else if (delta_H > 5e-3) {
            if (previous_direction == 1) delta /= 1.5;
            this->_epsilon = (1.0 / (1 + delta)) * this->_epsilon;
            previous_direction = -1;
          } else
            break;
          
          if (this->_epsilon > 1e7)
            throw std::runtime_error("Posterior is improper. Please check your model.");
          if (this->_epsilon == 0)
            throw std::runtime_error("No acceptably small step size could be found. Perhaps the posterior is not continuous?");
          
        }
        
        this->_z.copy_base(z_init);
        
      }
      
      sample transition(sample& init_sample) {
        
        this->seed(init_sample.cont_params(), init_sample.disc_params());
        
        try {
          init_stepsize();
        } catch (std::runtime_error e) {
          if (_err_stream) *_err_stream << e.what() << std::endl;
          return sample(init_sample.cont_params(), init_sample.disc_params(), 0, 0);
        }
        
        this->_epsilon *= 0.1;
        
        // Sample a large kinetic energy
        double sigma = sqrt( 0.5 * _z.q.size() );
        double T_threshold = sigma * (_n_threshold_sigma + sigma);
        double T = 0;
        
        while (T < T_threshold) { 
          this->_hamiltonian.sample_p(this->_z, this->_rand_int);
          T = this->_hamiltonian.T(this->_z); 
        }
        
        // Force particle away from local mode
        this->_hamiltonian.init(this->_z);
        double V_dot = this->_z.g.dot( this->_hamiltonian.dtau_dp(this->_z) );
        if (V_dot < 0) this->_z.p *= -1;
        
        this->_z.p *= _p_scale;
 
        // Evolve trajectory until second apex
        double V = this->_hamiltonian.V(this->_z);
        double V_max = V;
        double V_min = V;
        double V_old = V;
        
        std::vector<double> q_min = this->_z.q;
        std::vector<double> q_old = this->_z.q;
        
        V_dot = this->_z.g.dot( this->_hamiltonian.dtau_dp(this->_z) );
        double V_dot_old = V_dot;
        
        std::cout << V << "\t" << V_dot << std::endl;
        
        bool ignore = V_dot > 0 ? true : false;
        
        for (int i = 0; i < _max_iterations; ++i) {
          
          this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
          
          V_dot = this->_z.g.dot( this->_hamiltonian.dtau_dp(this->_z) );
          V = this->_hamiltonian.V(this->_z);
          
          std::cout << V << "\t" << V_dot << "\t" << this->_hamiltonian.H(this->_z) << std::endl;
          
          // NaN failure
          if (V == std::numeric_limits<double>::infinity() || V != V) {

            if (V_old < V_min) {
              V_min = V_old;
              q_min = q_old;
            }
            
            break;
          }
      
          if (V < V_min) {
            V_min = V;
            q_min = this->_z.q;
          }
          
          if (_sign(V_dot) != _sign(V_dot_old)) {
            
            if (V > V_max) {
              V_max = V;
              
              if (ignore) {
                ignore = false;
                continue;
              }
              else break;

            }
            
          }
          
          V_old = V;
          V_dot_old = V_dot;
          
        }
        
        return sample(q_min, init_sample.disc_params(), -V_min, 1);
        
      }
      
      void set_n_threshold_sigma(int n) { if (n >= 0) _n_threshold_sigma = n; }
      void set_p_scale(double a) { _p_scale = a; }
      void set_max_iterations(int n)  { if (n >= 0) _max_iterations = n; }

    protected:
      
      unit_e_point _z;
      expl_leapfrog<unit_e_metric<M, BaseRNG>, unit_e_point> _integrator;
      unit_e_metric<M, BaseRNG> _hamiltonian;
      
      BaseRNG& _rand_int;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;
      
      double _epsilon;
      
      int _n_threshold_sigma;
      double _p_scale;
      int _max_iterations;
      
      int _sign(double x) { return x > 0 ? 1 : -1; }
      
    };
    
  } // mcmc
  
} // stan

#endif
