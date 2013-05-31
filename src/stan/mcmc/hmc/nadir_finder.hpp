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
      _epsilon(1.0)
      {};
      
      void seed(const std::vector<double>& q, const std::vector<int>& r) {
        _z.q = q;
        _z.r = r;
      }
      
      void init_stepsize() {
        
        ps_point z_init(this->_z);
        
        while (1) {
          
          this->_z.copy_base(z_init);
          
          this->_hamiltonian.sample_p(this->_z, this->_rand_int);
          this->_hamiltonian.init(this->_z);
          
          double H0 = this->_hamiltonian.H(this->_z);
          
          this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
          
          double h = this->_hamiltonian.H(this->_z);
          if (h != h) h = std::numeric_limits<double>::infinity();
          
          double delta_H = fabs((H0 - h) / H0);
          
          std::cout << "delta = " << delta_H << std::endl;
          
          if      (delta_H < 5e-4)
            this->_epsilon = 2.0 * this->_epsilon;
          else if (delta_H > 5e-3)
            this->_epsilon = 0.5 * this->_epsilon;
          else
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
        
        std::cout << init_sample.cont_params().at(0) << "\t" << init_sample.cont_params().at(1) << std::endl;
        
        try {
          init_stepsize();
        } catch (std::runtime_error e) {
          if (_err_stream) *_err_stream << e.what() << std::endl;
          return sample(init_sample.cont_params(), init_sample.disc_params(), 0, 0);
        }
        
        std::cout << "Stepsize = " << this->_epsilon << std::endl;
        
        // Sample a large kinetic energy
        double sigma = sqrt( 0.5 * _z.q.size() );
        double T_threshold = sigma * (5 + sigma);
        double T = 0;
        
        while (T < T_threshold) { 
          this->_hamiltonian.sample_p(this->_z, this->_rand_int);
          T = this->_hamiltonian.T(this->_z); 
        }
        
        // Force particle away from local mode
        this->_hamiltonian.init(this->_z);
        double V_dot = this->_z.g.dot( this->_hamiltonian.dtau_dp(this->_z) );
        if (V_dot < 0) this->_z.p *= -1;
 
        // Evolve trajectory until second apex
        double V = this->_hamiltonian.V(this->_z);
        double V_max = V;
        double V_min = V;
        std::vector<double> q_min = this->_z.q;
        
        V_dot = this->_z.g.dot( this->_hamiltonian.dtau_dp(this->_z) );
        double V_dot_old = V_dot;
        
        bool ignore = V_dot > 0 ? true : false;
        
        while (1) {
          
          this->_integrator.evolve(this->_z, this->_hamiltonian, this->_epsilon);
          
          V_dot = this->_z.g.dot( this->_hamiltonian.dtau_dp(this->_z) );
          V = this->_hamiltonian.V(this->_z);
          
          std::cout << V << std::endl;
          
          // NaN failure
          if (V == std::numeric_limits<double>::infinity() || V != V) {
            break;
          }
          // New nadir
          else if (V_dot >= 0 & V_dot_old <= 0) {
            
            if (V < V_min) {
              V_min = V;
              q_min = this->_z.q;
            }
            
          }
          // New apex
          else if (V_dot <= 0 & V_dot_old >= 0) {
            
            if (V > V_max) {
              V_max = V;
              
              if (ignore) {
                ignore = false;
                continue;
              }
              else {
                break;
              }
            }
            
          }
          
          V_dot_old = V_dot;
          
        }
        
        std::cout << "Vmin = " << V_min << std::endl;
        
        return sample(q_min, init_sample.disc_params(), -V_min, 1);
        
      }

    protected:
      
      unit_e_point _z;
      expl_leapfrog<unit_e_metric<M, BaseRNG>, unit_e_point> _integrator;
      unit_e_metric<M, BaseRNG> _hamiltonian;
      
      BaseRNG& _rand_int;
      
      // Uniform(0, 1) RNG
      boost::uniform_01<BaseRNG&> _rand_uniform;
      
      double _epsilon;
      
    };
    
  } // mcmc
  
} // stan

#endif
