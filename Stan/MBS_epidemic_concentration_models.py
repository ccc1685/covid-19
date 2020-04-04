#save compartment models here as classes
#text description + Stan C++ code as string + networkx + maybe latex
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import networkx as nx

class model1a:
    def __init__(self):
        self.descript = """
        About: \n
        SIR model - expects I,R,D; sums R and D columns \n
        fits I and RD
        """
        print(self.descript)
        #initialize model specific parameters expected by Stan code
        self.stan_data = {
            'n_theta':2,
            'n_difeq':3,
            'n_ostates':3
            }
        self.math = """
        \begin{align} 
        \dot{dS} &= -\beta S I \\
        \dot{dI} &= \beta S I - \gamma I\\
        \dot{dR} &= \gamma I
         \end{align}
        """
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

                // priors for 1978 boarding school flu from Anastasia Chatzilena
                // theta[1] ~ lognormal(0,1);
                // theta[2] ~ gamma(0.004,0.02);  //Assume mean infectious period = 5 days 


                //likelihood
                for (i in 1:n_obs){
                    lambda[i,1] = y_hat[i,1]*n_pop; //convert to counts (ODE is normalized) 
                    lambda[i,2] = y_hat[i,3]*n_pop;


                    target += poisson_lpmf(y[i,1]|lambda[i,1]);
                    target += poisson_lpmf(recovered_death[i,1]|lambda[i,2]);
                    
                    //target += neg_binomial_2_lpmf(y[i,1]|lambda[i,1],0.9);
                    //target += neg_binomial_2_lpmf(recovered_death[i,1]|lambda[i,2],0.9);
                    
                    
                }
                //y[:,1] ~ poisson(lambda[:,1]);
                //recovered_death[:,1] ~ poisson(lambda[:,2]);
            }

            generated quantities {
                real R_0;      // Basic reproduction number
                R_0 = theta[1]/theta[2];
            }
        """
    def plotnetwork(self):
        print("S: susceptible")
        print("I: infected")
        print("RD: recovered_dead")

        G = nx.DiGraph()
        G.add_node('S')
        G.add_node('I')
        G.add_node('RD')

        G.add_edges_from([('S','I'),('I','RD')])
        nx.draw(G,with_labels=True)
        return
    

        
class model1b:
    def __init__(self):
        self.descript = """
        About: \n
        SIR model - expects I,R,D; ignores R and D\n
        fits I only
        """
        print(self.descript)
        #initialize model specific parameters expected by Stan code
        self.stan_data = {
            'n_theta':2,
            'n_difeq':3,
            'n_ostates':3
            }
        self.math = """
        \begin{align} 
        \dot{dS} &= -\beta S I \\
        \dot{dI} &= \beta S I - \gamma I\\
        \dot{dR} &= \gamma I
         \end
         """
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
              int y[n_obs,n_ostates];           // data, per-day-tally [cases]
              real t0;                // initial time point 
              real ts[n_obs];         // time points that were observed
            }

            transformed data {
              real x_r[0];
              int x_i[0];
              }

            parameters {
                real<lower = 0> theta[n_theta]; // model parameters 
                real<lower = 0, upper = 1> S0;  // initial fraction of susceptible individuals
            }

            transformed parameters{
                real y_hat[n_obs, n_difeq]; // solution from the ODE solver
                real y_init[n_difeq];     // initial conditions for fractions

                y_init[1] = S0; //S
                y_init[2] = 1-S0; //I
                y_init[3] = 0; //R

                y_hat = integrate_ode_rk45(SIR, y_init, t0, ts, theta, x_r, x_i);

            }

            model {
                real lambda[n_obs]; //poisson parameter [cases, recovered_dead]

                //priors
                S0 ~ beta(2, 2); //some prior for between 0 and 1 fraction of the population
                //for (i in 1:n_theta){
                //    theta[i] ~ lognormal(0,1);
                //}

                // examples from Anastasia Chatzilena
                theta[1] ~ lognormal(0,1);
                theta[2] ~ gamma(0.004,0.02);  //Assume mean infectious period = 5 days 


                //likelihood
                for (i in 1:n_obs){
                    lambda[i] = y_hat[i,2]*n_pop;
                    //target += poisson_lpmf(y[i]|lambda[i]);
                    }
                    y[:,1] ~ poisson(lambda);
            }

              generated quantities {
              real R_0;      // Basic reproduction number
              R_0 = theta[1]/theta[2];
              }
        """
    def plotnetwork(self):
        print("S: susceptible")
        print("I: infected")
        print("RD: recovered_dead")

        G = nx.DiGraph()
        G.add_node('S')
        G.add_node('I')
        G.add_node('RD')

        G.add_edges_from([('S','I'),('I','RD')])
        nx.draw(G,with_labels=True)
        return
    
class model2:
    def __init__(self):
        self.descript = """
        About: \n
        SICRDq model \n
        Some unknown I and known C, both go to recovered and death \n
        I and C have same leak to r and d but different to total infected Z
        """
        print(self.descript)
        #initialize model specific parameters expected by Stan code
        self.stan_data = {
            'n_theta':5,
            'n_difeq':5,
            'n_ostates':3
            }
#         self.math = """
#         \begin{eqnarray}
#         \frac{dC}{dt} &=& \sigma_c I - (\sigma_r + \sigma_d) C &\qquad\qquad &  
#         \frac{dD}{dt} &=& \sigma_d C\\
#         \frac{dR}{dt} &=& \sigma_r C &\qquad\qquad & 
#         \frac{dI}{dt} &=& \beta (I+qC) S - (\sigma_c  + \sigma_r + \sigma_d) I \\
#         \frac{dZ}{dt} &=& - \beta (I+qC) S  &\qquad\qquad & 
#         \end{eqnarray}
#         """
        self.stan = """
        functions {
            real[] SIR(real t,  // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters 
            real[] x_r,
            int[] x_i) {

            real du_dt[5];

            real sigmac = theta[1];
            real sigmar = theta[2];
            real sigmad =  theta[3];
            real q = theta[4]; 
            real beta = theta[5];
            real N = 1;
            
            real I = u[4];  # unknown infected
            real C = u[1];  # cases
            real S = u[5];  # susceptible

            du_dt[1] = sigmac*I - (sigmar + sigmad)*C; // dC  
            du_dt[2] = sigmad*C; // dD  
            du_dt[3] = sigmar*C; // dR  
            du_dt[4] = beta*N*(I+q*C)*S - (sigmac + sigmar + sigmad)*I; // dI  
            du_dt[5] = -beta*N*(I+q*C)*S; // dS
            
            return du_dt;
          }
        }

        data {
          int<lower = 1> n_obs;       // number of days observed
          int<lower = 1> n_theta;     // number of model parameters
          int<lower = 1> n_difeq;     // number of differential equations for yhat
          int<lower = 1> n_ostates;     // number of observed states
          int<lower = 1> n_pop;       // population
          real<lower = 1> n_scale;       // scale to match observed scale
          int y[n_obs,n_ostates];           // data, per-day-tally [cases,recovered,death]
          real t0;                // initial time point 
          real ts[n_obs];         // time points that were observed
          int max_num_steps; 
        }

        transformed data {
            real x_r[0];
            int x_i[0];           
        }

        parameters {
            real<lower = 0, upper = 1> S0;  // initial fraction of susceptible individuals
            real<lower = 0> theta[5]; // model parameters 
            
            
            //
        }

        transformed parameters{
            real u[n_obs, n_difeq]; // solution from the ODE solver
            real u_init[5];     // initial conditions for fractions
           
            // yhat for model is larger than y observed
            // also y initialized are not the same as y observed
            // y observed are cases (C), recovered (R), and deaths (D)
            // y init are latent infected (I), cases (C), and latent susceptible (S)

            u_init[1] = 0; //C
            u_init[2] = 0; //D 
            u_init[3] = 0; //R 
            u_init[4] = 1-S0;///n_pop; // I
            u_init[5] = S0;//S0*n_pop/n_scale; // 
            
            
            u = integrate_ode_rk45(SIR, u_init, t0, ts, theta, x_r, x_i);
           //u = integrate_ode_bdf(SIR, u_init, t0, ts, theta, x_r, x_i);

        }

        model {
            real lambda[n_obs,3]; //poisson parameter [cases, deaths, recovered]

            //priors
            S0 ~ beta(2, 2); //some prior for between 0 and 1 fraction of the population
            theta[1] ~ lognormal(0,0.1); //sigmac
            theta[2] ~ lognormal(0,0.1); //sigmar
            theta[3] ~ lognormal(0,0.1); //sigmad
            theta[4] ~ lognormal(0,0.1); //q
            theta[5] ~ lognormal(0,0.25); //beta
            //theta[5] ~ normal(n_scale,0.1*n_scale);
            
            

            //likelihood
            for (i in 1:n_obs){
                lambda[i,1] = u[i,1]*n_scale; //cases
                lambda[i,2] = u[i,3]*n_scale; //recovered
                lambda[i,3] = u[i,2]*n_scale; //dead
                
                //target += poisson_lpmf(y[i,1]|lambda[i,1]);
                //target += poisson_lpmf(y[i,2]|lambda[i,2]);
                //target += poisson_lpmf(y[i,3]|lambda[i,3]);

                target += poisson_lpmf(y[i,1]|max([lambda[i,1],0.0]));
                target += poisson_lpmf(y[i,2]|max([lambda[i,2],0.0]));
                target += poisson_lpmf(y[i,3]|max([lambda[i,3],0.0]));

                //y[i,1] ~ poisson(lambda[i,1]);
                //y[i,2] ~ poisson(lambda[i,2]);
                //y[i,3] ~ poisson(lambda[i,3]);
            }

        }

        generated quantities {
            real R_0;      // Basic reproduction number
            R_0 = (theta[4])/(theta[1]+theta[2]+theta[3]);
        }
        """
    def plotnetwork(self):
        print("S: susceptible")
        print("I: infected")
        print("C: identified cases")
        print("R: recovered")
        print("D: dead")
        G = nx.DiGraph()
        G.add_node('S')
        G.add_node('I')
        G.add_node('C')
        G.add_node('R')
        G.add_node('D')
        G.add_node('Ru')
        G.add_node('Du')
        G.add_edges_from([('S','I'),('I','C'),('I','Ru'),('I','Du'),('C','R'),('C','D')])
        nx.draw(G,with_labels=True)
        return
    
