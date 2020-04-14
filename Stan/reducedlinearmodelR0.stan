functions {
            real[] SIR(real t,  // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters
            real[] x_r,
            int[] x_i) {
            real du_dt[5];

            real R0 = theta[1];
            real sigmac = theta[2];
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real sigmau = theta[5];
            real q = theta[6];
            real mbase = theta[7];
            real mlocation = theta[8];

            real sigma = sigmar + sigmad;
            real beta = R0*sigma*(sigmac +sigmau)/(sigma+q*sigmac);

            real I = u[1];  // infected, latent
            real C = u[2];  // cases, observed

            beta *= mbase + (1-mbase)/(1 + exp(.2*(t - mlocation)));  // mitigation

            du_dt[1] = beta*(I+q*C) - sigmac*I - sigmau*I; //I
            du_dt[2] = sigmac*I - sigma*C;       //C
            du_dt[3] = beta*(I+q*C);                       //N_I
            du_dt[4] = sigmac*I; // N_C case appearance rate
            du_dt[5] = C; // cumulative C

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
          //real rel_tol;               // relative tolerance for ODE solver
          //real max_num_steps;         // for ODE solver
          }

        transformed data {
            real x_r[0];
            int x_i[0];
        }

        parameters {
            real<lower = 0> theta[n_theta]; // model parameters
        }

        transformed parameters{
            real u[n_obs, n_difeq];   // solution from the ODE solver
            real u_init[n_difeq];     // initial conditions for fractions

            real R0 = theta[1];
            real sigmac = theta[2];
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real sigmau = theta[5];
            real q = theta[6];
            real mbase = theta[7];
            real mlocation = theta[8];

            real sigma = sigmar + sigmad;
            real beta = R0*sigma*(sigmac +sigmau)/(sigma+q*sigmac);

            real lambda[n_obs,3]; //poisson parameter [cases, deaths, recovered]

            real cinit = y[1,1]/n_scale;

            u_init[1] = max([(beta-sigmau-sigmac)/max([sigmac,.001])*cinit, cinit]);  // I set from C
            u_init[2] = cinit;    //C  from data
            u_init[3] = u_init[1];        // N_I cumulative infected
            u_init[4] = cinit;        // N_C total cumulative cases
            u_init[5] = cinit;        // integral of active C

          //  print(theta)
          //  print(u_init)
            u = integrate_ode_rk45(SIR, u_init, t0, ts, theta, x_r, x_i,1e-3,1e-3,2000);

            lambda[1,1] = (u[1,4]-u_init[4])*n_scale; //C: cases per day
            lambda[1,2] = sigmar*(u[1,5]-u_init[5])*n_scale; //R: recovered per day
            lambda[1,3] = sigmad*(u[1,5]-u_init[5])*n_scale; //D: dead per day

            for (i in 2:n_obs){
                lambda[i,1] = (u[i,4]-u[i-1,4])*n_scale; //C: cases per day
                lambda[i,2] = sigmar*(u[i,5]-u[i-1,5])*n_scale; //R: recovered rate per day
                lambda[i,3] = sigmad*(u[i,5]-u[i-1,5])*n_scale; //D: dead rate per day
            }
        }

        model {
            //priors
            theta[1] ~ gamma(1.8,1.8/2.6);    //R0
            theta[2] ~ gamma(1.5,1.5/.2);   //sigmac
            theta[3] ~ gamma(2.,20);  //sigmar
            theta[4] ~ gamma(2.,20);  //sigmad
            theta[5] ~ gamma(2.,20);  //sigmau
            theta[6] ~ exponential(2.);         //q
            theta[7] ~ beta(2.,5.);  //mbase
            theta[8] ~ lognormal(log(tm+5),.3); //mlocation

            //likelihood
            //lambda[1,1] =  sigma_c * I for day
            //lambda[1,2] =  sigma_r * C for day
            //lambda[1,3] =  sigma_d * C for day

            target += poisson_lpmf(max(y[1,1],0)|max([lambda[1,1],1.0])); //C
            target += poisson_lpmf(max(y[1,2],0)|max([lambda[1,2],1.0])); //R
            target += poisson_lpmf(max(y[1,3],0)|max([lambda[1,3],1.0])); //D

            for (i in 2:n_obs){
                target += poisson_lpmf(max(y[i,1],0)|max([lambda[i,1],1.0])); //C
                target += poisson_lpmf(max(y[i,2],0)|max([lambda[i,2],1.0])); //R
                target += poisson_lpmf(max(y[i,3],0)|max([lambda[i,3],1.0])); //D
            }
        }

        generated quantities {
            real car[n_obs];
            real ifr[n_obs];

            //u[3]:NI
            //u[4]:NC
            //u[5]: integrated C

            real ll_; // log-likelihood for model

            //likelihood
            //R0 = beta*(sigmar+sigmad+q*sigmac)/((sigmar+sigmad)*(sigmac+sigmau));
            //real sigma = sigmar + sigmad;
            //beta = R0*sigma*(sigmac +sigmau)/(sigma+q*sigmac);

            car[1] = u[1,4]/u[1,3];         //total cases / total infected
            ifr[1] = sigmad*u[1,5]/u[1,3];  // total dead / total infected

            ll_ = poisson_lpmf(max(y[1,1],0)|max([lambda[1,1],1.0]));
            ll_ += poisson_lpmf(max(y[1,2],0)|max([lambda[1,2],1.0]));
            ll_ += poisson_lpmf(max(y[1,3],0)|max([lambda[1,3],1.0]));

            for (i in 2:n_obs){
                car[i] = u[i,4]/u[i,3];
                ifr[i] = sigmad*u[i,5]/u[i,3];

                ll_ += poisson_lpmf(max(y[i,1],0)|max([lambda[i,1],1.0]));
                ll_ += poisson_lpmf(max(y[i,2],0)|max([lambda[i,2],1.0]));
                ll_ += poisson_lpmf(max(y[i,3],0)|max([lambda[i,3],1.0]));
            }
          //  print(ll_)
        }
