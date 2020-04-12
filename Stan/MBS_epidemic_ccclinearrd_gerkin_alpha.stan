functions {
            real[] SIR(real t,  // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters 
            real[] x_r,
            int[] x_i) {

            real du_dt[2];
        
            real beta = theta[1];
            real sigmac = theta[2];
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real q = theta[5]; 
            real f = theta[6]; 
            real mbase = theta[7]; 
            real mlocation = theta[8]; 
            real mrate = theta[9]; 
            real cmax = theta[10];
            real c50 = theta[11];
            
            real I = u[1];  # unknown infected
            real C = u[2];  # cases
            
            beta *= mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)));
            sigmac *= 1 + cmax*t/(c50+t);
            
            
            du_dt[1] = beta*(I+q*C) - sigmac*I - f*(sigmar+sigmad)*I; //I
            du_dt[2] = sigmac*I - (sigmar+sigmad)*C; //C
            
            return du_dt;
          }
        }

        data {
          int<lower = 1> n_obs;       // number of days observed
          int<lower = 1> n_theta;     // number of model parameters
          int<lower = 1> n_difeq;     // number of differential equations for yhat
          int<lower = 1> n_ostates;     // number of observed states
          real<lower = 1> n_scale;       // scale to match observed scale
          int y[n_obs,n_ostates];           // data, per-day-tally [cases,recovered,death]
          real t0;                // initial time point 
          real tm; //start day of mitigation
          real ts[n_obs];         // time points that were observed
          }

        transformed data {
            real x_r[0];
            int x_i[0];           
        }

        parameters {
            real<lower = 0> theta[n_theta]; // model parameters 
        }

        transformed parameters{
            real u[n_obs, n_difeq]; // solution from the ODE solver
            real u_init[n_difeq];     // initial conditions for fractions
            
            real beta = theta[1];
            real sigmac = theta[2];
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real q = theta[5]; 
            real f = theta[6]; 
            real mbase = theta[7]; 
            real mlocation = theta[8]; 
            real mrate = theta[9]; 
            real cmax = theta[10];
            real c50 = theta[11];
            real theta_init = theta[12];
            
            real sigmac2;
            
            real lambda[n_obs,3]; //poisson parameter [cases, deaths, recovered]

            
                        
            u_init[1] = theta_init/n_scale; // I
            u_init[2] = y[1,1]/n_scale; //C
                     
            //print(theta)
            u = integrate_ode_rk45(SIR, u_init, t0, ts, theta, x_r, x_i);
            
            //lambda
            sigmac2 = sigmac*(1 + cmax*1/(c50+1));
            lambda[1,1] = .5*(u[1,1]+u_init[1])*sigmac2*n_scale; //C: cases sigma_c*\int I
            lambda[1,2] = .5*(u[1,2]+u_init[2])*sigmar*n_scale; //R: recovered sigma_r*\int C
            lambda[1,3] = .5*(u[1,2]+u_init[2])*sigmad*n_scale; //D: dead sigma_d * \int C
        
            for (i in 2:n_obs){
                sigmac2 = sigmac*(1 + cmax*0.5*i/(c50+0.5*i));
                lambda[i,1] = .5*(u[i,1]+u[i-1,1])*sigmac2*n_scale; //C: cases sigma_c*\int_{interval} I
                lambda[i,2] = .5*(u[i,2]+u[i-1,2])*sigmar*n_scale; //R: recovered sigma_r*C
                lambda[i,3] = .5*(u[i,2]+u[i-1,2])*sigmad*n_scale; //D: dead
            
            }
  
        }

        model {
  
            //priors
            
            //for (i in 1:n_theta){theta[1] ~ lognormal(log(0.1),10);};
            //theta[5] ~ lognormal(log(0.25),10);
            
            
            //real beta = theta[1];
            //real sigmac = theta[2];
            //real sigmar = theta[3];
            //real sigmad =  theta[4];
            //real q = theta[5]; 
            //real f = theta[6]; 
            //real mbase = theta[7]; 
            //real mlocation = theta[8]; 
            //real mrate = theta[9]; 
            //real cmax = theta[10];
            //real c50 = theta[11];
            
            // [0.25,
            //0.1,
            //0.01,
            //0.01,
            //0.01,
            //1.0,
            //0.1,
            //1.0,
            //1.0,
            //0.1,
            //10.0,
            //1.0]

            theta[1] ~ lognormal(log(0.25),1); //beta 
            theta[2] ~ lognormal(log(0.1),1); //sigmac
            theta[3] ~ lognormal(log(0.01),1); //sigmar
            theta[4] ~ lognormal(log(0.01),1); //sigmad
            theta[5] ~ lognormal(log(0.01),1); //q
            theta[6] ~ lognormal(log(1),1); //f
            theta[7] ~ lognormal(log(0.1),1); //mbase  
            theta[8] ~ lognormal(log(tm),5); //mlocation 
            theta[9] ~ lognormal(log(1),5); //mrate
            theta[10] ~ lognormal(log(0.1),1);//cmax 
            theta[11] ~ lognormal(log(10),1);//c50 
            theta[12] ~ lognormal(log(1),0.1);// theta_init 

          
            //likelihood
            //lambda[1,1] = .5*(u[1,1]+u_init[1])*sigmac*n_scale; //C: cases sigma_c*\int I
            //lambda[1,2] = .5*(u[1,2]+u_init[2])*sigmar*n_scale; //R: recovered sigma_r*\int C
            //lambda[1,3] = .5*(u[1,2]+u_init[2])*sigmad*n_scale; //D: dead sigma_d * \int C
        
            target += poisson_lpmf(y[1,1]|max([lambda[1,1],0.0])); //C
            target += poisson_lpmf(y[1,2]|max([lambda[1,2],0.0])); //R
            target += poisson_lpmf(y[1,3]|max([lambda[1,3],0.0])); //D

            
            for (i in 2:n_obs){
                //lambda[i,1] = .5*(u[i,1]+u[i-1,1])*sigmac*n_scale; //C: cases sigma_c*\int_{interval} I
                //lambda[i,2] = .5*(u[i,2]+u[i-1,2])*sigmar*n_scale; //R: recovered sigma_r*C
                //lambda[i,3] = .5*(u[i,2]+u[i-1,2])*sigmad*n_scale; //D: dead
            
                target += poisson_lpmf(y[i,1]|max([lambda[i,1],0.0])); //C
                target += poisson_lpmf(y[i,2]|max([lambda[i,2],0.0])); //R
                target += poisson_lpmf(y[i,3]|max([lambda[i,3],0.0])); //D
            }

        }

        generated quantities {
        
            real ll_; // log-likelihood for model
         
            //likelihood
            ll_ = poisson_lpmf(y[1,1]|max([lambda[1,1],0.0]));
            ll_ += poisson_lpmf(y[1,2]|max([lambda[1,2],0.0]));
            ll_ += poisson_lpmf(y[1,3]|max([lambda[1,3],0.0]));

            
            for (i in 2:n_obs){
                ll_ += poisson_lpmf(y[i,1]|max([lambda[i,1],0.0]));
                ll_ += poisson_lpmf(y[i,2]|max([lambda[i,2],0.0]));
                ll_ += poisson_lpmf(y[i,3]|max([lambda[i,3],0.0]));
            }
            
            print(ll_)
         
        }
