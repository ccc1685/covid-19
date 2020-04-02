#save compartment models here as classes
#text description + Stan C++ code as string + networkx + maybe latex

class model1:
    def __init__(self):
        self.stan = """
            functions {
              real[] SIR(real t,  // time
              real[] y,           // system state {susceptible, infected, recovered}
              real[] theta,       // parameters 
              real[] x_r,
              int[] x_i) {

              real dy_dt[3];

              real beta = theta[1];
              real gamma = theta[2];

              real S = y[1];  # susceptible
              real I = y[2];  # infected
              real R = y[3];  # recovered

              dy_dt[1] = -beta*S*I; //dS  
                dy_dt[2] = beta*S*I - gamma*I; //dI
                dy_dt[3] = gamma*I; //dR  

              return dy_dt;
              }
            }

            data {
              int<lower = 1> n_obs;       // number of days observed
              int<lower = 1> n_theta;     // number of model parameters
              int<lower = 1> n_difeq;     // number of differential equations for yhat
              int<lower = 1> n_ostates;     // number of observed states
              int<lower = 1> n_pop;       // population 
              int y[n_obs,n_ostates];           // data, per-day-tally [cases, deaths, recovered]
              real t0;                // initial time point 
              real ts[n_obs];         // time points that were observed
            }

            transformed data {
                int recovered_death[n_obs,1];
                real x_r[0];
                int x_i[0];

                for (i in 1:n_obs){
                    recovered_death[i,1] = y[i,2] + y[i,3]; 
                }
            }

            parameters {
                real<lower = 0> theta[n_theta]; // model parameters 
                real<lower = 0, upper = 1> S0;  // initial fraction of susceptible individuals
            }

            transformed parameters{
                real y_hat[n_obs, n_difeq]; // solution from the ODE solver
                real y_init[3];     // initial conditions for fractions

                // yhat for model is larger than y observed
                // also y initialized are not the same as y observed
                // y observed are cases (C), recovered (R), and deaths (D)
                // y init are latent infected (I), cases (C), and latent susceptible (S)

                y_init[1] = S0; //S
                y_init[2] = 1-S0; //I
                y_init[3] = 0; //R

                y_hat = integrate_ode_rk45(SIR, y_init, t0, ts, theta, x_r, x_i);

            }

            model {
                real lambda[n_obs,2]; //poisson parameter [cases, recovered_dead]

                //priors
                S0 ~ beta(2, 2); //some prior for between 0 and 1 fraction of the population
                for (i in 1:n_theta){
                    theta[i] ~ lognormal(0,1);
                }

                // prior on ODE parameters left out, below are some examples from Anastasia Chatzilena
                // theta[1] ~ lognormal(0,1);
                // theta[2] ~ gamma(0.004,0.02);  //Assume mean infectious period = 5 days 


                //likelihood
                for (i in 1:n_obs){
                    lambda[i,1] = y_hat[i,1]*n_pop; //convert to counts (ODE is normalized ?) [lambda cases:1, lambda deaths:2, lambda recovered:3]
                    lambda[i,2] = y_hat[i,3]*n_pop;


                    //target += poisson_lpmf(y[i,1]|lambda[i,1]);
                    //target += poisson_lpmf(recovered_death[i,1]|lambda[i,2]);
                }
                y[:,1] ~ poisson(lambda[:,1]);
                recovered_death[:,1] ~ poisson(lambda[:,2]);
            }

            generated quantities {
                real R_0;      // Basic reproduction number
                R_0 = theta[1]/theta[2];
            }
        """
    