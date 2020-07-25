// SICRMQC2R2DX.stan
// Latent variable nonlinear SICR model with mitigation and release, q>0

functions {

         //  mitigation function
         real transition(real base,real location,real transition, real t) {
                 real scale;
                 if (base == 1)
                     scale = 1;
                 else
                    scale = base + (1. - base)/(1. + exp((t - location)/transition));
                 return scale;
         }

         // Relaxation function
         real relax(real base, real t) {
              return base*(1 + 0.42924175/(1 + exp(-0.2154182*(t - 20.29067964))));
         }


                // nonlinear SICR model with R delay ODE function
                   real[] SICR(
                   real t,             // time
                   real[] u,           // system state {infected,cases,susceptible}
                   real[] theta,       // parameters
                   real[] x_r,
                   int[] x_i
                   )
                   {
                     real du_dt[8];
                     real f1 = theta[1];          // beta - sigmau - sigmac
                     real f2 = theta[2];          // beta - sigma u
                     real sigmar = theta[3];
                     real sigmad =  theta[4];
                     real sigmau = theta[5];
                     real q = theta[6];
                     real mbase = theta[7];
                     real mlocation = theta[8];
                     real sigmar1 = theta[9];
                     real sigmad1 = theta[10];
                     real trelax = theta[11];
                     real cbase = theta[12];
                     real clocation = theta[13];
                     real ctransition = theta[14];
                     real mtransition = theta[15];
                     real minit;

                     real sigmac = f2/(1+f1);
                     real beta = f2 + sigmau;
                     real sigma = sigmar + sigmad;

                     real I = u[1];  // infected, latent
                     real C = u[2];  // cases, observed
                     real Z = u[3];  // total infected
                     real R1 = u[5]; // recovery compartment 1
                     real D1 = u[6]; // death compartment 1

                     trelax += mlocation;
                     sigmac *= transition(cbase,clocation,ctransition,t);  // case detection change
                     if (t < trelax) {
                        //beta *= mitigation(mbase,mlocation,t);  // mitigation
                        beta *= transition(mbase,mlocation,mtransition,t);  // mitigation
                     }
                     else {
                        minit = transition(mbase,mlocation,mtransition,trelax);
                        beta *= relax(minit,t-trelax);   // relaxation from lockdown
                     }

                     du_dt[1] = beta*(I+q*(C+R1))*(1-Z) - sigmac*I - sigmau*I; //I
                     du_dt[2] = sigmac*I - sigma*C;                            //C
                     du_dt[3] = beta*(I+q*(C+R1))*(1-Z);                       //N_I
                     du_dt[4] = sigmac*I;                    // N_C case appearance rate
                     du_dt[5] = sigmar*C - sigmar1*R1;       // recovery compartment 1
                     du_dt[6] = sigmad*C - sigmad1*D1;       // death compartment 1
                     du_dt[7] = sigmar1*R1;                  // total case recoveries
                     du_dt[8] = sigmad1*D1;                  // total case deaths

                     return du_dt;
                   }
        }


#include data.stan

transformed data {
    real x_r[0];
    int x_i[0];
    int n_difeq = 5;     // number of differential equations for yhat
    real mtransition = 7.;
    real ctransition = 21.;
    //real q = 0.;
    //real n_pop = 1000;
    //real cbase = 1.;
    //real clocation = 10.;
}

parameters {
    real<lower=0> f1;             // initial infected to case ratio
    real<lower=0> f2;             // f2  beta - sigmau
    real<lower=0> sigmar;         // recovery rate
    real<lower=0> sigmad;         // death rate
    real<lower=0> sigmar1;      // 1st compartment recovery rate
    real<lower=0> sigmad1;      // 1st compartment death rate
    real<lower=0> sigmau;         // I disappearance rate
    real<lower=0> mbase;          // mitigation strength
    real<lower=0> mlocation;      // day of mitigation application
    real<lower=0> trelax;         // day of relaxation from mitigation
    real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
    real<lower=0> extra_std_R;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
    real<lower=0> extra_std_D;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
    real<lower=0> q;              // infection factor for cases
    real<lower=0> cbase;          // case detection factor
    real<lower=0> clocation;      // day of case change
  real<lower=1> n_pop;      // population size

}

transformed parameters{
  real lambda[n_total,3];     //neg_binomial_2 rate [new cases, new recovered, new deaths]
  real car[n_total];          //total cases / total infected
  real ifr[n_total];          //total dead / total infected
  real Rt[n_total];           // time dependent reproduction number
  real phi[3];

  real u_init[8];     // initial conditions for fractions

  real sigmac = f2/(1+f1);
  real beta = f2 + sigmau;
  real sigma = sigmar + sigmad;
  real R0 = beta*(sigma+q*sigmac)/sigma/(sigmac+sigmau);   // reproduction number


  phi[1] = max([1/(extra_std^2),1e-10]); // likelihood over-dispersion of std for C
  phi[2] = max([1/(extra_std_R^2),1e-10]); // likelihood over-dispersion of std for R
  phi[3] = max([1/(extra_std_D^2),1e-10]); // likelihood over-dispersion of std for D

  {
     real theta[15] = {f1,f2,sigmar,sigmad,sigmau,q,mbase,mlocation,sigmar1,sigmad1,trelax,cbase,clocation,ctransition,mtransition};

     real u[n_total, 8];   // solution from the ODE solver
     real sigmact;
     real betat;

     real cinit = y[1,1]/n_pop;
     u_init[1] = f1*cinit;      // I set from f1 * C initial
     u_init[2] = cinit;         // C from data
     u_init[3] = u_init[1];     // N_I cumulative infected
     u_init[4] = cinit;         // N_C total cumulative cases
     u_init[5] = sigmar*cinit;  // R1
     u_init[6] = sigmad*cinit;  // D1
     u_init[7] = 0;             // total R
     u_init[8] = 0;             // total D

     u = integrate_ode_rk45(SICR, u_init, ts[1]-1, ts, theta, x_r, x_i,1e-3,1e-5,2000);

     for (i in 1:n_total){
        car[i] = u[i,4]/u[i,3];
        ifr[i] = u[i,8]/u[i,3];
        betat = beta*transition(mbase,mlocation,mtransition,i)*(1-u[i,3]);
        sigmact = sigmac*transition(cbase,clocation,ctransition,i);
        Rt[i] = betat*(sigma+q*sigmact)/sigma/(sigmact+sigmau);
        }

     lambda[1,1] = max([(u[1,4]-u_init[4])*n_pop,0.01]); //C: cases per day
     lambda[1,2] = max([(u[1,7]-u_init[7])*n_pop,0.01]); //R: recovered per day
     lambda[1,3] = max([(u[1,8]-u_init[8])*n_pop,0.01]); //D: dead per day

     for (i in 2:n_total){
        lambda[i,1] = max([(u[i,4]-u[i-1,4])*n_pop,0.01]); //C: cases per day
        lambda[i,2] = max([(u[i,7]-u[i-1,7])*n_pop,0.01]); //R: recovered per day
        lambda[i,3] = max([(u[i,8]-u[i-1,8])*n_pop,0.01]); //D: dead per day
        }
    }
}


model {
//priors Stan convention:  gamma(shape,rate), inversegamma(shape,rate)

    f1 ~ gamma(2.,.1);                     // f1  initital infected to case ratio
    f2 ~ gamma(100.,350.);                 // f2  beta - sigmau
    sigmar ~ gamma(10.,3.);              // sigmar
    sigmad ~ gamma(10.,3.);              // sigmad
    sigmar1 ~ exponential(4.);             // sigmar1
    sigmad1 ~ exponential(4.);             // sigmad1
    sigmau ~ exponential(2.);              // sigmau
    q ~ exponential(2.);                   // q
    mbase ~ exponential(3.);               // mbase
    mlocation ~ normal(tm,20.);     // mlocation
    extra_std ~ exponential(1.);          // likelihood over dispersion std C
    extra_std_R ~ exponential(1.);          // likelihood over dispersion std R
    extra_std_D ~ exponential(1.);          // likelihood over dispersion std D
    cbase ~ exponential(.2);               // cbase
    clocation ~ normal(120.,60.);    // clocation
    n_pop ~ normal(4e6,2e6);        // population
    trelax ~ normal(70,30.);     // trelax


//#include likelihood.stan
// Likelihood function

for (i in 1:n_obs){
  for (j in 1:3) {
    if (y[i,j] > -1.)
      target += neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi[j]);
    }
}

}

//#include generatedquantities.stan
// generated quantities

generated quantities {
    real llx[n_obs, 3];
    real ll_; // log-likelihood for model
    int n_data_pts;
    real y_proj[n_total,3];

    ll_ = 0;
    n_data_pts = 0;
    for (i in 1:n_obs) {
        for (j in 1:3) {
           if (y[i,j] > -1.){
                llx[i, j] = neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi[j]);
                n_data_pts += 1;
                ll_ += llx[i, j];
               }
           else {
                llx[i,j] = 0.;
               }
          }
    }

    for (i in 1:n_total) {
        for (j in 1:3) {
          if (lambda[i,j] < 1e8)
            y_proj[i,j] = neg_binomial_2_rng(lambda[i,j],phi[j]);
          else
            y_proj[i,j] = poisson_rng(1e8);
        }
    }
}
