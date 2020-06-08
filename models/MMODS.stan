// MMODS.stan
// Latent variable nonlinear SICR model with mitigation from mobile data, q>0, and fixed population

functions {
         // time transition functions for beta and sigmac
         real mobility(real base,real t) {
                 real scale;
                    scale = base + (1 - base) / (1 + exp(0.47 * (t - 32))/(1 + exp(.52*(t-40.))));
                    //scale = 0.55 + (1 - 0.55) / (1 + exp(0.47 * (t - 32))/(1 + exp(.52*(t-40.))));

                 return scale;
         }

         // nonlinear SICR model ODE function
           real[] SICR(
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

             real sigma = sigmar + sigmad;
             real sigmac = f2/(1+f1);
             real beta = f2 + sigmau;

             real I = u[1];  // infected, latent
             real C = u[2];  // cases, observed
             real Z = u[3];  // total infected

             //sigmac *= transition(cbase,clocation,t);  // case detection change
             beta *= mobility(mbase,t);  // mitigation

             du_dt[1] = beta*(I+q*C)*(1-Z) - sigmac*I - sigmau*I; // I
             du_dt[2] = sigmac*I - sigma*C;                       // C
             du_dt[3] = beta*(I+q*C)*(1-Z);                       // Z = N_I
             du_dt[4] = sigmac*I;                                 // N_C case appearance rate
             du_dt[5] = C;                                        // integrated C

             return du_dt;
            }
        }

data {
  int<lower=1> n_obs;       // number of days observed
  int<lower=1> n_ostates;   // number of observed states
  int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
  real tm;                    // start day of mitigation
  real ts[n_obs];             // time points that were observed
  }


transformed data {
    real x_r[0];
    int x_i[0];
    int n_difeq = 5;     // number of differential equations for yhat
    // real q = 0.;
    real n_pop = 100000;
}

parameters {
    real<lower=0> f1;             // initial infected to case ratio
    real<lower=0> f2;             // f2  beta - sigmau
    real<lower=0> sigmar;         // recovery rate
    real<lower=0> sigmad;         // death rate
    real<lower=0> sigmau;         // I disappearance rate
    real<lower=0> mbase;          // mitigation strength
    //real<lower=0> mlocation;      // day of mitigation application
    real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
    real<lower=0> q;              // infection factor for cases
}

transformed parameters{
  real<lower=.01> lambda[n_obs,3]; //neg_binomial_2 rate [new cases, new recovered, new deaths]
  real car[n_obs];      //total cases / total infected
  real ifr[n_obs];      //total dead / total infected
  real Rt[n_obs];           // time dependent reproduction number

  real u_init[5];     // initial conditions for fractions
  real u_end[5];       // last point

  real sigmac = f2/(1+f1);
  real beta = f2 + sigmau;
  real sigma = sigmar + sigmad;
  real R0 = beta*(sigma+q*sigmac)/sigma/(sigmac+sigmau);   // reproduction number
  real phi = max([1/(extra_std^2),1e-10]); // likelihood over-dispersion of std

  {
     real theta[7] = {f1, f2, sigmar, sigmad, sigmau, q, mbase};
     real u[n_obs, 5];   // solution from the ODE solver
     real betat;
     real sigmact;

     real cinit = y[1,1]/n_pop;

     u_init[1] = f1*cinit;      // I set from f1 * C initial
     u_init[2] = cinit;         //C  from data
     u_init[3] = u_init[1];     // N_I cumulative infected
     u_init[4] = cinit;         // N_C total cumulative cases
     u_init[5] = cinit;         // integral of active C

     u = integrate_ode_rk45(SICR, u_init, ts[1]-1, ts, theta, x_r, x_i,1e-2,1e-2,2000);

     for (j in 1:5){
      u_end[j] = u[n_obs,j];
     }

     for (i in 1:n_obs){
        car[i] = u[i,4]/u[i,3];
        ifr[i] = sigmad*u[i,5]/u[i,3];
        betat = beta*mobility(mbase,i)*(1-u[i,3]);
        sigmact = sigmac;
        Rt[i] = betat*(sigma+q*sigmact)/sigma/(sigmact+sigmau);
        }

     lambda[1,1] = max([(u[1,4]-u_init[4])*n_pop,1.0]); //C: cases per day
     lambda[1,2] = max([sigmar*(u[1,5]-u_init[5])*n_pop,1.0]); //R: recovered per day
     lambda[1,3] = max([sigmad*(u[1,5]-u_init[5])*n_pop,1.0]); //D: dead per day

     for (i in 2:n_obs){
        lambda[i,1] = max([(u[i,4]-u[i-1,4])*n_pop,1.0]); //C: cases per day
        lambda[i,2] = max([sigmar*(u[i,5]-u[i-1,5])*n_pop,1.0]); //R: recovered rate per day
        lambda[i,3] = max([sigmad*(u[i,5]-u[i-1,5])*n_pop,1.0]); //D: dead rate per day
        }

    }
}


model {
    //priors Stan convention:  gamma(shape,rate), inversegamma(shape,rate)
    f1 ~ gamma(2.,1./10.);                 // f1  initital infected to case ratio
    f2 ~ gamma(1.5,1.);                    // f2  beta - sigmau
    sigmar ~ inv_gamma(4.,.2);             // sigmar
    sigmad ~ inv_gamma(2.78,.185);         // sigmad
    sigmau ~ inv_gamma(2.3,.15);           // sigmau
    q ~ exponential(2.);                   // q
    mbase ~ exponential(1.);               // mbase
    //mlocation ~ lognormal(log(tm+5),1.);   // mlocation
    extra_std ~ exponential(1.);           // likelihood over dispersion std

//likelihood
for (i in 1:n_obs){
  for (j in 1:3) {
    if (y[i,j] > 0.)
      target += neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi);
    }
}

}

generated quantities {
    real llx[n_obs, 3];
    real ll_; // log-likelihood for model
    int n_data_pts;
    int n_proj;
    real y_proj[n_ostates];
    real u[1,5];

    real times[1] = {184.};
    real theta[7] = {f1, f2, sigmar, sigmad, sigmau, q, mbase};
    n_proj = 184; // project out six months May 15 - Nov 15

    ll_ = 0.;
    n_data_pts = 0;
    for (i in 1:n_obs) {
        for (j in 1:3) {
           if (y[i,j] > 0.){
                llx[i, j] = neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi);
                n_data_pts += 1;
                ll_ += llx[i, j];
               }
           else {
                llx[i,j] = 0.;
                }
        }
    }


    u = integrate_ode_rk45(SICR, u_end, 0.,times, theta, x_r, x_i,1e-2,1e-2,2000);

    y_proj[1] = u[1,3]*n_pop;                   //cumulative projected mean infected
    y_proj[2] = sigmad*u[1,5]*n_pop;           // cumulative projected mean dead

}
