functions {
            real[] SIR(
            real t,             // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters
            real[] x_r,
            int[] x_i
            )
            {
            real du_dt[5];

            real f1 = theta[1];
            real f2 = theta[2];
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real sigmau = theta[5];
            real q = theta[6];
            real mbase = theta[7];
            real mlocation = theta[8];

            real sigmac = f2/(1+f1);
            real beta = f2 + sigmau;
            real sigma = sigmar + sigmad;

            real I = u[1];  // infected, latent
            real C = u[2];  // cases, observed

            beta *= mbase + (1-mbase)/(1 + exp(.2*(t - mlocation)));  // mitigation

            du_dt[1] = beta*(I+q*C) - sigmac*I - sigmau*I; //I
            du_dt[2] = sigmac*I - sigma*C;       //C
            du_dt[3] = beta*(I+q*C);                       //N_I
            du_dt[4] = sigmac*I; // N_C case appearance rate
            du_dt[5] = C; // integrated C

            return du_dt;
          }
        }

        data {
          int<lower = 1> n_obs;       // number of days observed
          int<lower = 1> n_theta;     // number of model parameters
          int<lower = 1> n_difeq;     // number of differential equations for yhat
          int<lower = 1> n_ostates;   // number of observed states
          real<lower = 1> n_scale;    // scale to match observed scale
          int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
          real t0;                    // initial time point
          real tm;                    // start day of mitigation
          real ts[n_obs];             // time points that were observed
          //real rel_tol;             // relative tolerance for ODE solver
          //real max_num_steps;       // for ODE solver
          }

        transformed data {
            real x_r[0];
            int x_i[0];
        }

        parameters {
            real<lower = 0> theta[n_theta]; // model parameters
        }

        transformed parameters{

          real lambda[n_obs,3]; //neg_binomial_2 rate [new cases, new recovered, new deaths]
          real car[n_obs];      //total cases / total infected
          real ifr[n_obs];      //total dead / total infected

          real f1 = theta[1];
          real f2 = theta[2];
          real sigmar = theta[3];
          real sigmad =  theta[4];
          real sigmau = theta[5];
          real q = theta[6];
          real mbase = theta[7];
          real mlocation = theta[8];

          real u_init[n_difeq];     // initial conditions for fractions

          real sigmac = f2/(1+f1);
          real beta = f2 + sigmau;
          real sigma = sigmar + sigmad;
          real R0 = beta*(sigma+q*sigmac)/sigma/(sigmac+sigmau);   // reproduction number

          real phi = 1/(theta[9]^2);  // likelihood over dispersion std

          {
            real u[n_obs, n_difeq];   // solution from the ODE solver

            real cinit = y[1,1]/n_scale;

            u_init[1] = f1*cinit;      // I set from f1 * C initial
            u_init[2] = cinit;         //C  from data
            u_init[3] = u_init[1];     // N_I cumulative infected
            u_init[4] = cinit;         // N_C total cumulative cases
            u_init[5] = cinit;         // integral of active C

            //print(theta)
            //print(u_init)

            u = integrate_ode_rk45(SIR, u_init, t0, ts, theta, x_r, x_i,1e-3,1e-3,2000);

            car[1] = u[1,4]/u[1,3];
            ifr[1] = sigmad*u[1,5]/u[1,3];

            lambda[1,1] = (u[1,4]-u_init[4])*n_scale; //C: cases per day
            lambda[1,2] = sigmar*(u[1,5]-u_init[5])*n_scale; //R: recovered per day
            lambda[1,3] = sigmad*(u[1,5]-u_init[5])*n_scale; //D: dead per day

            for (i in 2:n_obs){
                car[i] = u[i,4]/u[i,3];
                ifr[i] = sigmad*u[i,5]/u[i,3];

                lambda[i,1] = (u[i,4]-u[i-1,4])*n_scale; //C: cases per day
                lambda[i,2] = sigmar*(u[i,5]-u[i-1,5])*n_scale; //R: recovered rate per day
                lambda[i,3] = sigmad*(u[i,5]-u[i-1,5])*n_scale; //D: dead rate per day
                }
            }
        }

        model {
            //priors Stan convention:  gamma(shape,rate), inversegamma(shape,rate)
            theta[1] ~ gamma(1.5,1.5);           // f1  initital infected to case ratio
            theta[2] ~ gamma(1.5,4.5);           // f2  beta - sigmau
            theta[3] ~ inv_gamma(4.,.2);             // sigmar
            theta[4] ~ inv_gamma(2.78,.185);             // sigmad
            theta[5] ~ inv_gamma(2.78,.185);             // sigmau
            theta[6] ~ exponential(2.);          // q
            theta[7] ~ exponential(3.);          // mbase
            theta[8] ~ lognormal(log(tm+5),.5);  // mlocation
            theta[9] ~ exponential(1.);          // likelihood over dispersion std

            //likelihood
            //lambda[1,1] =  sigma_c * I for day
            //lambda[1,2] =  sigma_r * C for day
            //lambda[1,3] =  sigma_d * C for day

            target += neg_binomial_2_lpmf(max(y[1,1],0)|max([lambda[1,1],1.0]),phi); //C
            target += neg_binomial_2_lpmf(max(y[1,2],0)|max([lambda[1,2],1.0]),phi); //R
            target += neg_binomial_2_lpmf(max(y[1,3],0)|max([lambda[1,3],1.0]),phi); //D

            for (i in 2:n_obs){
                target += neg_binomial_2_lpmf(max(y[i,1],0)|max([lambda[i,1],1.0]),phi); //C
                target += neg_binomial_2_lpmf(max(y[i,2],0)|max([lambda[i,2],1.0]),phi); //R
                target += neg_binomial_2_lpmf(max(y[i,3],0)|max([lambda[i,3],1.0]),phi); //D
            }
        }

        generated quantities {

            real ll_; // log-likelihood for model

            ll_ = neg_binomial_2_lpmf(max(y[1,1],0)|max([lambda[1,1],1.0]),phi);
            ll_ += neg_binomial_2_lpmf(max(y[1,2],0)|max([lambda[1,2],1.0]),phi);
            ll_ += neg_binomial_2_lpmf(max(y[1,3],0)|max([lambda[1,3],1.0]),phi);
            for (i in 2:n_obs){
                ll_ += neg_binomial_2_lpmf(max(y[i,1],0)|max([lambda[i,1],1.0]),phi);
                ll_ += neg_binomial_2_lpmf(max(y[i,2],0)|max([lambda[i,2],1.0]),phi);
                ll_ += neg_binomial_2_lpmf(max(y[i,3],0)|max([lambda[i,3],1.0]),phi);
            }
          //  print(ll_)
        }
