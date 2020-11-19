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
real<lower=0> sigmau;             // uninfected rate
real<lower=0> sigmac;             // case rate
real<lower=0> sigmar;             // recovery rate
real<lower=0> sigmad;                  // death rate

}

transformed parameters {

}

model {

  target += exponential_lpdf(beta | 1.);
  target += exponential_lpdf(sigmar | 1.);
  target += exponential_lpdf(sigmad | 5.);

  {
    real dI;
    real I;
    real S;
    real C;
    S = N;
    C = y[1,1];
    I = C*100;

    for (i in 1:n_obs){
      dI = beta*S*I/N;
      S -=  dI;
      I += beta*S*I/N - sigmac*I - sigmau*I;
      C += sigmac*I - sigmar*C - sigmad*C;

  
      print(I)
      print(C)

      target += poisson_lpmf(y[i,1] | max([sigmac*I,.0001]));
      target += poisson_lpmf(y[i,2] | max([sigmar*C,.0001]));
      target += poisson_lpmf(y[i,3] | max([sigmad*C,.0001]));
    }
}

}
