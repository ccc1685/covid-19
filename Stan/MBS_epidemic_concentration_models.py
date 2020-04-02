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

                // prior on ODE parameters left out, below are some examples from Anastasia Chatzilena
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
        """
        print(self.descript)
        #initialize model specific parameters expected by Stan code
        self.stan_data = {
            'n_theta':5,
            'n_difeq':5,
            'n_ostates':3
            }
        self.math = """
        \begin{eqnarray}
        \frac{dC}{dt} &=& \sigma_c I - (\sigma_r + \sigma_d) C &\qquad\qquad &  
        \frac{dD}{dt} &=& \sigma_d C\\
        \frac{dR}{dt} &=& \sigma_r C &\qquad\qquad & 
        \frac{dI}{dt} &=& \beta (I+qC) (1-Z) - (\sigma_c  + \sigma_r + \sigma_d) I \\
        \frac{dZ}{dt} &=& \beta (I+qC) (1-Z)  &\qquad\qquad & 
        \end{eqnarray}
        """
        self.stan = """
        functions {
          real[] SIR(real t,  // time
          real[] y,           // system state {infected,cases,susceptible}
          real[] theta,       // parameters 
          real[] x_r,
          int[] x_i) {

          real dy_dt[5];

          real sigmac = theta[1];
          real sigmar = theta[2];
          real sigmad =  theta[3];
          real beta = theta[4]; 
          real q = theta[5]; 

          real I = y[1];  # infected
          real C = y[2];  # cases
          real Z = y[3];  # susceptible


            dy_dt[1] = sigmac*I - (sigmar + sigmad)*C; //dC  
            dy_dt[2] = sigmad*C; //dD 
            dy_dt[3] = sigmar*C; //dR  
            dy_dt[4] = beta*(I+q*C)*(1-Z) - (sigmac + sigmar + sigmad)*I; //dI 
            dy_dt[5] = beta*(I+q*C)*(1-Z); //dZ 

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
            real x_r[0];
            int x_i[0];
        }

        parameters {
            real<lower = 0> theta[n_theta]; // model parameters 
            real<lower = 0, upper = 1> S0;  // initial fraction of susceptible individuals
        }

        transformed parameters{
            real y_hat[n_obs, n_difeq]; // solution from the ODE solver
            real y_init[5];     // initial conditions for fractions

            // yhat for model is larger than y observed
            // also y initialized are not the same as y observed
            // y observed are cases (C), recovered (R), and deaths (D)
            // y init are latent infected (I), cases (C), and latent susceptible (S)

            y_init[1] = 1/n_pop; //I, ccc has other formulation
            //y_init[1] = S0/n_pop + (theta[2]+theta[3])/theta[1]*log(1-S0/n_pop) + 1/n_pop;
            y_init[2] = 0; //C
            y_init[3] = S0; //S
            y_init[4] = 0; // dummy
            y_init[5] = 0; // dummy

            y_hat = integrate_ode_rk45(SIR, y_init, t0, ts, theta, x_r, x_i);

        }

        model {
            real lambda[n_obs,3]; //poisson parameter [cases, deaths, recovered]

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
                lambda[i,2] = y_hat[i,2]*n_pop;
                lambda[i,3] = y_hat[i,3]*n_pop;

                target += poisson_lpmf(y[i,1]|lambda[i,1]);
                target += poisson_lpmf(y[i,2]|lambda[i,2]);
                target += poisson_lpmf(y[i,3]|lambda[i,3]);

                //y[i,1] ~ poisson(lambda[i,1]);
                //y[i,2] ~ poisson(lambda[i,2]);
                //y[i,3] ~ poisson(lambda[i,3]);
            }

        }

        generated quantities {
            real R_0;      // Basic reproduction number
            R_0 = theta[4]/(theta[1]+theta[5]);
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
    
class model3:
    def __init__(self):
        self.descript = """
        About: \n
        """
        print(self.descript)
        #initialize model specific parameters expected by Stan code
        self.stan_data = {
            'n_theta':5,
            'n_difeq':6,
            'n_ostates':3
            }
        self.math = """
        \begin{eqnarray}
        \frac{dC}{dt} &=& \sigma_c I - \gamma_r C - \gamma_d C &\qquad\qquad &  
        \frac{dD}{dt} &=& \gamma_d C\\
        \frac{dR}{dt} &=& \gamma_r C &\qquad\qquad & 
        \frac{dI}{dt} &=& \beta I (1-Z) - \sigma_c I - \sigma_r I \\
        \frac{dZ}{dt} &=& \beta I (1-Z)  &\qquad\qquad & \frac{dU}{dt} &=& \sigma_r I
        \end{eqnarray}
        """
        self.stan = """
            functions {
              real[] SIR(real t,  // time
                            real[] y,           // system state {infected,cases,susceptible}
                            real[] theta,       // parameters 
                            real[] x_r,
                            int[] x_i)
                {
                    real dy_dt[6];

                    real sigmac = theta[1];
                    real gammar = theta[2];
                    real gammad =  theta[3];
                    real beta = theta[4]; 
                    real sigmar = theta[5];
                    real I = y[1];  # infected
                    real C = y[2];  # cases
                    real Z = y[3];  # susceptible

                    dy_dt[1] = sigmac*I - gammar*C - gammad*C; //dC
                    dy_dt[2] = gammad*C; //dD
                    dy_dt[3] = gammar*C; //dR
                    dy_dt[4] = beta*I*(1-Z) - sigmac*I - sigmar*I; //dI
                    dy_dt[5] = beta*I*(1-Z); //dZ, Carson rewrite for S
                    dy_dt[6] = sigmar*I; //dU
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
                real mut0;                // prior mean for initial time point 
                real ts[n_obs];         // time points that were observed
            }

            transformed data {
                real x_r[0];
                int x_i[0];
            }

            parameters {
                real<lower = 0> theta[n_theta]; // model parameters 
                real<lower = 0, upper = 1> S0;  // initial fraction of susceptible individuals
                real t0; // start date for epidemic (local spread), in days relative to data day indices, can be negative
            }

            transformed parameters{
                real y_hat[n_obs, n_difeq]; // solution from the ODE solver
                real y_init[3];     // initial conditions for fractions

                // yhat for model is larger than y observed
                // also y initialized are not the same as y observed
                // y observed are cases (C), recovered (R), and deaths (D)
                // y init are latent infected (I), cases (C), and latent susceptible (S)

                y_init[1] = 1/n_pop; //I, ODE system assumes fractional quantities ?
                y_init[2] = 0; //C
                y_init[3] = S0; //S

                y_hat = integrate_ode_rk45(SIR, y_init, t0, ts, theta, x_r, x_i);
            }

            model {
                real lambda[n_obs,3]; //poisson parameter [cases, deaths, recovered]

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
                    lambda[i,2] = y_hat[i,2]*n_pop;
                    lambda[i,3] = y_hat[i,3]*n_pop;

                    target += poisson_lpmf(y[i,1]|lambda[i,1]);  # cases
                    target += poisson_lpmf(y[i,2]|lambda[i,2]);  # deaths
                    target += poisson_lpmf(y[i,3]|lambda[i,3]);  # recovered

                    //y[i,1] ~ poisson(lambda[i,1]);
                    //y[i,2] ~ poisson(lambda[i,2]);
                    //y[i,3] ~ poisson(lambda[i,3]);
                }

            }

            generated quantities {
                real R_0;      // Basic reproduction number
                R_0 = theta[4]/(theta[1]+theta[5]);
            }
        """
    def plotnetwork(self):
        print("S: susceptible")
        print("I: infected")
        print("U: unknown cases")
        print("C: identified cases")
        print("R: recovered cases")
        print("D: dead cases")
        G = nx.DiGraph()
        G.add_node('S')
        G.add_node('I')
        G.add_node('C')
        G.add_node('R')
        G.add_node('D')
        G.add_node('U')
        G.add_edges_from([('S','I'),('I','C'),('C','R'),('C','D')])
        nx.draw(G,with_labels=True)
        return