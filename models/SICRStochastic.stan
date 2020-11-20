// SICRStochastic.stan
// stochastic SIR model


data {
      int<lower=1> n_obs;       // number of days observed
      int<lower=0> n_weeks;
      int n_total;               // total number of days simulated, n_total-n_obs is days beyond last data point
      int<lower=1> n_ostates;   // number of observed states
      int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
      real tm;                    // start day of mitigation
      real ts[n_total];             // time points that were observed + projected
  }

transformed data {
  int N = 100000;
  int y_wk[n_obs,n_weeks];
  for (k in 1:n_ostates){
    for (i in 1:n_weeks){
      y_wk[i,k] = y[7*(i-1) + 1,k];
      for (j in 2:7)
        y_wk[i,k] += y[7*(i-1) + j,k];
    }
  }
}


parameters {

real<lower=0> beta[n_weeks];             // infection rate
real<lower=0> lambda[n_weeks];             // infection rate
real<lower=0> sigmau;             // uninfected rate
real<lower=0> sigmac;             // case rate
real<lower=0> sigmac0;             // case rate
real<lower=0> sigmar;             // recovery rate
real<lower=0> sigmad;                  // death rate
real<lower=0> alpha;

}

transformed parameters {

}

model {

  sigmac ~ exponential(1.);
  sigmac ~ exponential(1.);
  sigmar ~ exponential(2.);
  sigmad ~ exponential(4.);
  alpha ~ exponential(5.);


  {
    real dI;
    real I;
    real C;

    C = y[1,1];
    I = C*10;

    for (i in 1:n_weeks){

    lambda[i] ~ cauchy(0.,10.);
    beta[i] ~ normal(0.,lambda[i]);

      I += beta[i]*I/N - sigmac*I - sigmau*I + alpha;
      C += (sigmac*i* sigmac0)*I - sigmar*C - sigmad*C;

      target += poisson_lpmf(y_wk[i,1] | max([sigmac*I,.0001]));
      target += poisson_lpmf(y_wk[i,2] | max([sigmar*C,.0001]));
      target += poisson_lpmf(y_wk[i,3] | max([sigmad*C,.0001]));
    }
}

}
