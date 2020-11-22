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
  int y_wk[n_weeks,n_ostates];
  print(n_weeks);
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
//real<lower=0> lambda[n_weeks];  // infection rate
real<lower=0> sigmau;             // uninfected rate
real<lower=0> sigmac;             // case rate
real<lower=0> sigmar;             // recovery rate
real<lower=0> sigmad;             // death rate
real<lower=0> alpha;
}

transformed parameters {
  real I;
  real C;
  real dI[n_weeks];
  real dC[n_weeks];
  real dR[n_weeks];
  real dD[n_weeks];

  C = 0;
  I = 0;
  for (i in 1:n_weeks){
    I += alpha;
    I *= exp(beta[i] - sigmac - sigmau);
    dC[i] = sigmac*I;
    C += dC[i];
    C *= exp(-(sigmar+sigmad));
    dI[i] = (exp(beta[i])-1)*I + alpha;
    dR[i] = sigmar*C;
    dD[i] = sigmad*C;
  }
}

model {
  sigmac ~ exponential(2.);
  //sigmac0 ~ exponential(4.);
  sigmau ~ exponential(5.);
  sigmar ~ exponential(4.);
  sigmad ~ exponential(8.);
  alpha ~ exponential(10.);

    for (i in 1:n_weeks){
      //lambda[i] ~ cauchy(0.,1.);
      beta[i] ~ exponential(.5);
      target += poisson_lpmf(y_wk[i,1] | dC[i]);
      target += poisson_lpmf(y_wk[i,2] | dR[i]);
      target += poisson_lpmf(y_wk[i,3] | dD[i]);
    }
    for (i in 2:n_weeks-1){
      target += normal_lpdf(beta[i+1]-beta[i] | 0, .05);
      target += normal_lpdf(beta[i+1]-2*beta[i]+beta[i-1] | 0, .5);
    }
}

// generated quantities

generated quantities {

    real car[n_weeks];
    real ifr[n_weeks];
    real Rt[n_weeks];
    int y_proj[n_weeks*7,n_ostates];
    real C_cum;
    real I_cum;
    real D_cum;

    C_cum = 0;
    I_cum = 0;
    D_cum = 0;

    for (i in 1:n_weeks) {
      C_cum += dC[i];
      I_cum += dI[i];
      D_cum += dD[i];
      car[i] = C_cum/I_cum;
      ifr[i] = D_cum/I_cum;
      Rt[i] = beta[i]/sigmac;
      for (j in 1:7){
        y_proj[7*(i-1) + j,1] = poisson_rng(min([dC[i]/7,1e8]));
        y_proj[7*(i-1) + j,2] = poisson_rng(min([dR[i]/7,1e8]));
        y_proj[7*(i-1) + j,3] = poisson_rng(min([dD[i]/7,1e8]));
      }
    }
}
