// SIRStochastic.stan
// stochastic SIR model


data {
      int<lower=1> n_obs;       // number of days observed
      int n_total;               // total number of days simulated, n_total-n_obs is days beyond last data point
      int<lower=1> n_ostates;   // number of observed states
      int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
      real tm;                    // start day of mitigation
      real ts[n_total];             // time points that were observed + projected
  }

transformed data {
  int N = 100000;
}


parameters {

real<lower=0> beta;             // infection rate
real<lower=0> sigmar;             // recovery rate
real<lower=0> sigmad;                  // death rate

}

transformed parameters {


}

model {

  beta ~ exponential(1.);
  sigmac ~ exponential(1.);
  sigmad ~ exponential(5.);

  {

  real<lower=0> dI;
  real<lower=0> I;
  real<lower=0> S;

  S = N;
  I = 1;
    for (i in 1:n_obs){
      dI = beta*S*I/N;
      I +=  dI - y[2,i] - y[3,i];
      S -=  dI;
      target += Poisson_lpdf(y[2,i] | sigmar*I) + poisson_ldf(y[3,i] | sigmad*I);
    }
}

}
