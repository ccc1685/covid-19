// SICRLMQC2R.stan
// Latent variable linear SICR model with mitigation, q>0, sigmac time dependence, delayed R

#include functionsSICRL2R.stan
#include data.stan

transformed data {
    real x_r[0];
    int x_i[0];
    // real q = 0.;
    real n_pop = 1000;
}

parameters {
    real<lower=0> f1;             // initial infected to case ratio
    real<lower=0> f2;             // f2  beta - sigmau
    real<lower=0> sigmar;         // recovery rate
    real<lower=0> sigmad;         // death rate
    real<lower=0> sigmau;         // I disappearance rate
    real<lower=0> extra_std;      // phi = 1/extra_std^2 in neg_binomial_2(mu,phi)
    real<lower=0> mbase;          // mitigation strength
    real<lower=0> mlocation;      // day of mitigation application
    real<lower=0> q;              // infection factor for cases
    real<lower=0> cbase;          // case detection factor
    real<lower=0> clocation;      // day of case change
    real<lower=0> sigmar1;        // 1st compartment recovery rate
    //real<lower=1> n_pop;        // population size
}

#include transformedparametersL2R.stan
// model block for all SICR models
model {
    //priors Stan convention:  gamma(shape,rate), inversegamma(shape,rate)
    f1 ~ gamma(2.,1./10.);                 // f1  initital infected to case ratio
    f2 ~ gamma(1.5,1.);                    // f2  beta - sigmau
    sigmar ~ inv_gamma(4.,.2);             // sigmar
    sigmad ~ inv_gamma(2.78,.185);         // sigmad
    sigmau ~ inv_gamma(2.3,.15);           // sigmau
    extra_std ~ exponential(1.);           // likelihood over dispersion std
    q ~ exponential(1.);                   // q
    mbase ~ exponential(1.);               // mbase
    mlocation ~ lognormal(log(tm+5),1.);   // mlocation
    cbase ~ exponential(.2);               // cbase
    clocation ~ lognormal(log(20.),2.);    // clocation
    //n_pop ~ lognormal(log(1e5),4.);        // population
    sigmar1 ~ inv_gamma(4.,.2);            // sigmar1

    // Likelihood function

    for (i in 1:n_obs){
      for (j in 1:3) {
        if (y[i,j] > 0.)
          target += neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi);
        }
    }
}
#include generatedquantities.stan
