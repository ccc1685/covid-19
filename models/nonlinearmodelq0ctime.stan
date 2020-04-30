// Latent variable SIR model with q=0 (i.e. perfect quarantine of cases)
// include time dependence in sigmac to model change in case criteria or differences in testing


functions { // time transition functions for beta and sigmac
                real transition(real base,real location,real t) {
                   return base + (1-base)/(1 + exp(.2*(t - location)));
                }
            // SIR model
            real[] SIR(
            real t,             // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters
            real[] x_r,
            int[] x_i
            )
            {
            real du_dt[5];
            real f1 = theta[1];          // beta - sigmau - sigmac
            real f2 = theta[2];          // beta - sigma u
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real sigmau = theta[5];
            real q = theta[6];
            real mbase = theta[7];
            real mlocation = theta[8];
            real cbase = theta[9];
            real clocation = theta[10];

            real sigma = sigmar + sigmad;
            real sigmac = f2/(1+f1);
            real beta = f2 + sigmau;

            real I = u[1];  // infected, latent
            real C = u[2];  // cases, observed
            real Z = u[3];  // total infected

            sigmac *= cbase + (1-cbase)/(1 + exp(.2*(t - clocation)));  // case detection change
            //beta *= mbase + (1-mbase)/(1 + exp(.2*(t - mlocation)));  // mitigation

            du_dt[1] = beta*(I+q*C)*(1-Z) - sigmac*I - sigmau*I; //I
            du_dt[2] = sigmac*I - sigma*C;       //C
            du_dt[3] = beta*(I+q*C)*(1-Z);                       //N_I
            du_dt[4] = sigmac*I; // N_C case appearance rate
            du_dt[5] = C; // integrated C

            return du_dt;
          }
        }

        data {
          int<lower=1> n_obs;         // number of days observed
          int<lower=1> n_difeq;       // number of differential equations for yhat
          int<lower=1> n_ostates;     // number of observed states
          real<lower=1> n_scale;      // scale variables for numerical stability
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
            real q = 0.;
        }

        parameters {
          // model parameters
            real<lower=0> f1;       //initital infected to case ratio
            real<lower=0> f2;       // f2  beta - sigmau
            real<lower=0> sigmar;   // recovery rate
            real<lower=0> sigmad;   // death rate
            real<lower=0> sigmau;   // I disapperance rate
            //real<lower=0> q;              // infection factor for cases
            //real<lower=0> mbase;          // mitigation factor
            //real<lower=0> mlocation;      // day of mitigation application
            real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
            real<lower=0> cbase;          // case detection factor
            real<lower=0> clocation;      // day of case change
            real<lower=1> n_pop;      // population size
        }

        transformed parameters{
            real lambda[n_obs,3];     //neg_binomial_2 rate [new cases, new recovered, new deaths]
            real car[n_obs];          //total cases / total infected
            real ifr[n_obs];          //total dead / total infected
            real Rt[n_obs];           // time dependent reproduction number

            real u_init[n_difeq];     // initial conditions for odes
            real sigmac = f2/(1+f1);
            real beta = f2 + sigmau;
            real sigma = sigmar + sigmad;
            real R0 = beta*(sigma+q*sigmac)/sigma/(sigmac+sigmau);   // reproduction number
            //real Rlast =R0 * (mbase + (1-mbase)/(1 + exp(.2*(n_obs - mlocation))));  // R(t=last day)
            real phi = 1/(extra_std^2);  // likelihood over-dispersion of std

          {  // local
             real theta[10] = {f1,f2,sigmar,sigmad,sigmau,q,1.,1.,cbase,clocation};
             real u[n_obs, n_difeq];   // solution from the ODE solver
             real sigmact;

             real cinit = y[1,1]/n_pop;

             u_init[1] = f1*cinit;      // I set from f1 * C initial
             u_init[2] = cinit;         // C  from data
             u_init[3] = u_init[1];     // N_I cumulative infected
             u_init[4] = cinit;         // N_C total cumulative cases
             u_init[5] = cinit;         // integral of active C

            //print(theta)
            //print(u_init)

             u = integrate_ode_rk45(SIR, u_init, t0, ts, theta, x_r, x_i,1e-3,1e-3,2000);

             car[1] = u[1,4]/u[1,3];
             ifr[1] = sigmad*u[1,5]/u[1,3];
             sigmact = sigmac*transition(cbase,clocation,1);
             Rt[1] = beta*(sigma+q*sigmact)/sigma/(sigmact+sigmau);

             lambda[1,1] = (u[1,4]-u_init[4])*n_pop; //C: cases per day
             lambda[1,2] = sigmar*(u[1,5]-u_init[5])*n_pop; //R: recovered per day
             lambda[1,3] = sigmad*(u[1,5]-u_init[5])*n_pop; //D: dead per day

             for (i in 2:n_obs){
                car[i] = u[i,4]/u[i,3];
                ifr[i] = sigmad*u[i,5]/u[i,3];
                sigmact = sigmac*transition(cbase,clocation,i);
                Rt[i] = beta*(sigma+q*sigmact)/sigma/(sigmact+sigmau);

                lambda[i,1] = (u[i,4]-u[i-1,4])*n_pop; //C: cases per day
                lambda[i,2] = sigmar*(u[i,5]-u[i-1,5])*n_pop; //R: recovered rate per day
                lambda[i,3] = sigmad*(u[i,5]-u[i-1,5])*n_pop; //D: dead rate per day
                }
            }
        }

        model {
            // Stan convention:  gamma(shape,rate), inversegamma(shape,rate)
            f1 ~ gamma(2.,1./10.);                 // f1  initital infected to case ratio
            f2 ~ gamma(1.5,1.);                    // f2  beta - sigmau
            sigmar ~ inv_gamma(4.,.2);             // sigmar
            sigmad ~ inv_gamma(2.78,.185);         // sigmad
            sigmau ~ inv_gamma(2.3,.15);           // sigmau
            //q ~ exponential(10.);                // q
            //mbase ~ exponential(4.);               // mbase
            //mlocation ~ lognormal(log(tm+5),1.);   // mlocation
            extra_std ~ exponential(1.);           // likelihood over dispersion std
            cbase ~ exponential(.2);               // mbase
            clocation ~ lognormal(log(20.),2.);   // mlocation
            n_pop ~ lognormal(log(1e5),2.); // population

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
            real llx[n_obs, 3];

            ll_ = 0;
            for (i in 1:n_obs) {
                for (j in 1:3) {
                    llx[i, j] = neg_binomial_2_lpmf(max(y[i,j],0)|max([lambda[i,j],1.0]),phi);
                    ll_ += llx[i, j];
                    }
                }
            }
