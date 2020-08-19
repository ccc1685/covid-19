// SICRMQCR2HV.stan
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
                     real du_dt[9];
                     real f1 = theta[1];          // beta - sigmau - sigmac
                     real f2 = theta[2];          // beta - sigma u
                     real sigmaVS = theta[3];
                     real sigmaSR = theta[4];
                     real sigmaSR1 = theta[5];
                     real sigmaIV = theta[6];
                     real sigmaRI = theta[7];
                     real sigmaDI = theta[8];
                     real sigmaRC = theta[9];
                     real sigmaRH1 = theta[10];
                     real sigmaRH2 = theta[11];
                     real sigmaRV = theta[12];
                     real sigmaH1C = theta[13];
                     real sigmaH1V = theta[14];
                     real sigmaH2H1 = theta[15];
                     real sigmaRH1 = theta[16];
                     real sigmaDH1 = theta[17];

                     real q = theta[18];
                     real mbase = theta[19];
                     real mlocation = theta[20];

                     real trelax = theta[23];
                     real cbase = theta[24];
                     real clocation = theta[25];
                     real ctransition = theta[26];
                     real mtransition = theta[27];
                     real minit;



                     real sigmaCI = f2/(1+f1);
                     real beta = f2 + sigmaRI + sigmaDI;


                     real S = u[1];  // susceptibles, latent
                     real I = u[2];  // infected, latent
                     real C = u[2];  // cases, observed
                     real R = u[4];  // recovery compartment from C, observed
                     real RI = u[5]; // recovery compartment from I, latent
                     real H1 = u[6]; // hospital compartment 1, observed
                     real H2 = u[7]; // hosplital compartment 2, observed
                     real V = u[8]


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

                     du_dt[1] = -beta*(I+q*C)*S -sigmaVS*S + sigmaSR*R + sigmaSR1*R1 ; //dS/dt
                     du_dt[2] = beta*(I+q*C)*S +sigmaIV*V - sigmaCI*I - sigmaRI*I - sigmaDI*I;     //dI/dt
                     du_dt[3] = sigmaCI*I - sigmaRC*C - sigmaDC*C - sigmaH1C*C;          //dC/dt
                     du_dt[4] = sigmaRC*C + sigmaRH1*H1 + sigmaRH2*H2 +sigmaRV*V - sigmaSR*R;     // dR/dt
                     du_dt[5] = sigmaRI*I - sigmaSR1*R1;                               // dRI/dt
                     du_dt[6] = sigmaH1C*C +sigmaH1V*V- sigmaH2H1*H1 - sigmaRH1*H1 - sigmaDH1*H1; // dH1/dt
                     du_dt[7] = sigmaH2H1*H1 - sigmaRH2*H2 - sigmaDH2*H2;             // dH2/dt
                     du_dt[8] = sigmaVS*S -sigmaIV*V - sigmaRV*V -sigmaH1V*V;   // dV/dt
                     du_dt[9] = sigmaVS*S;                                              // cum V

                     return du_dt;



                   }
        }


data {
      int<lower=1> n_obs;       // number of days observed
      int n_total;               // total number of days simulated, n_total-n_obs is days beyond last data point
      int<lower=1> n_ostates;   // number of observed states
      int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
      real tm;                    // start day of mitigation
      real ts[n_total];             // time points that were observed + projected
  }


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
    real<lower=0> f2;             // f2  beta - sigmaCI - sigmaDI

    real<lower=0> sigmaSR
    real<lower=0> sigmaIV
    real<lower=0> sigmaRI
    real<lower=0> sigmaDI
    real<lower=0> sigmaRC
    real<lower=0> sigmaRH1
    real<lower=0> sigmaRH2
    real<lower=0> sigmaRV
    real<lower=0> sigmaH1C
    real<lower=0> sigmaH1V
    real<lower=0> sigmaH2H1
    real<lower=0> sigmaRH1
    real<lower=0> sigmaDH1

    real<lower=0> q;              // infection factor for cases
    real<lower=0> mbase;          // mitigation strength
    real<lower=0> mlocation;      // day of mitigation application
    real<lower=0> trelax;         // day of relaxation from mitigation
//    real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
//    real<lower=0> extra_std_R;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
//    real<lower=0> extra_std_D;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)

    real<lower=0> cbase;          // case detection factor
    real<lower=0> clocation;      // day of case change
    real<lower=1> n_pop;      // population size

}

transformed parameters{
  real lambda[n_total,3];     //neg_binomial_2 rate [new cases, new recovered, new deaths]
  //real car[n_total];          //total cases / total infected
  //real ifr[n_total];          //total dead / total infected
  //real Rt[n_total];           // time dependent reproduction number
  //real phi[3];
  real frac_infectious[n_total];
  real cumulativeV[n_total];

  real u_init[9];     // initial conditions for fractions

  real sigmaCI = f2/(1+f1);
  real beta = f2 + sigmaRI + sigmaDI;

  //real R0 = beta*(sigma+q*sigmac)/sigma/(sigmac+sigmau);   // reproduction number


  //phi[1] = max([1/(extra_std^2),1e-10]); // likelihood over-dispersion of std for C
  //phi[2] = max([1/(extra_std_R^2),1e-10]); // likelihood over-dispersion of std for R
  //phi[3] = max([1/(extra_std_D^2),1e-10]); // likelihood over-dispersion of std for D

  {
     real theta[15] = {f1,f2,sigmar,sigmad,sigmau,q,mbase,mlocation,sigmar1,sigmad1,trelax,cbase,clocation,ctransition,mtransition};

     real u[n_total, 9];   // solution from the ODE solver

     real cinit = y[1,1]/n_pop;

     u_init[1] = 1;                 //S
     u_init[2] = f1*cinit;         // I, f1 * Cinit
     u_init[3] = cinit;            // C, from data
     u_init[4] = 0;             // R
     u_init[5] = 0;             // RI
     u_init[6] = 0;             // H1
     u_init[7] = 0;             // H2
     u_init[8] = 0;             // V
     u_init[9] = 0;             // cum V

     u = integrate_ode_rk45(SICR, u_init, ts[1]-1, ts, theta, x_r, x_i,1e-3,1e-5,2000);

     for (i in 1:n_total){
        //car[i] = u[i,4]/u[i,3];
        //ifr[i] = u[i,8]/u[i,3];
        //betat = beta*transition(mbase,mlocation,mtransition,i)*(1-u[i,3]);
        //sigmact = sigmac*transition(cbase,clocation,ctransition,i);
        //Rt[i] = betat*(sigma+q*sigmact)/sigma/(sigmact+sigmau);
        }

     for (i in 1:n_total){
        lambda[i,1] = sigmaCI*u[2]*n_pop;  //C: cases per day
        lambda[i,2] = sigmaRC*U[3] + sigmaRH1*U[6] + sigmaRH2*U[7] +sigmaRV*U[8]; // new R per day
        lambda[i,3] = sigmaDC*u[3] + sigmaDH1*u[6] + sigmaDH2*U[7] + sigma; //new D per day
        lambda[i,4] = sigmaH1C*u[3] + sigmaH1V*U[8]; //new H1 per day
        lambda[i,5] = sigmaH1V*U[8];                 //new H2 per day
        //lambda[i,6] = sigmaDC*u[3] + sigmaDH1*u[6] + sigmaDH2*U[7] + sigma; //new V per day
        frac_infectious[i] = u[i,2];
        cumulativeV[i] = u[i,9];
        }
    }
}


model {
//priors Stan convention:  exponential(rate), gamma(shape,rate), inversegamma(shape,rate)

    f1 ~ gamma(2.,.1);                     // f1  initital infected to case ratio
    f2 ~ gamma(100.,350.);                 // f2  beta - sigmau

    sigmaVS ~ exponential(7.);
    sigmaSR ~ exponential(180.);
    sigmaSR1 ~ exponential(180.);
    sigmaIV ~ exponential(100.);
    sigmaRI ~ exponential(14.);
    sigmaDI ~ exponential(30.);
    sigmaRC ~ exponential(14.);
    sigmaRH1 ~ exponential(14.);
    sigmaRH2 ~ exponential(14.);
    sigmaRV ~ exponential(31.);
    sigmaH1C ~ exponential(14.);
    sigmaH1V ~ exponential(30.);
    sigmaH2H1 ~ exponential(7.);
    sigmaRH1 ~ exponential(14.);
    sigmaDH1 ~ exponential(30.);
    q ~ exponential(2.);                   // q
    n_pop ~ normal(4e6,2e6);        // population

//    mbase ~ exponential(3.);               // mbase
//    mlocation ~ normal(tm,20.);     // mlocation
//    extra_std ~ exponential(1.);          // likelihood over dispersion std C
//    extra_std_R ~ exponential(1.);          // likelihood over dispersion std R
//    extra_std_D ~ exponential(1.);          // likelihood over dispersion std D
//    cbase ~ exponential(.2);               // cbase
//    clocation ~ normal(120.,60.);    // clocation
//    trelax ~ normal(70,30.);     // trelax


// Likelihood function

for (i in 1:n_obs){
  for (j in 1:5) {
    if (y[i,j] > -1.)
      target += poisson_lpmf(y[i,j]|lambda[i,j]);
    }
}

}

// generated quantities

generated quantities {
    real llx[n_obs, 5];
    real ll_; // log-likelihood for model
    int n_data_pts;
    real y_proj[n_total,5];

    ll_ = 0;
    n_data_pts = 0;
    for (i in 1:n_obs) {
        for (j in 1:5) {
           if (y[i,j] > -1.){
                llx[i, j] = poisson_lpmf(y[i,j]|lambda[i,j]);
                n_data_pts += 1;
                ll_ += llx[i, j];
               }
           else {
                llx[i,j] = 0.;
               }
          }
    }

    for (i in 1:n_total) {
        for (j in 1:5) {
          if (lambda[i,j] < 1e8)
            y_proj[i,j] = poisson_rng(lambda[i,j]);
          else
            y_proj[i,j] = poisson_rng(1e8);
        }
    }
}
