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
real<lower=0> sigmac0;             // case rate
real<lower=0> sigmac1;             // case rate
real<lower=0> sigmac2;
//real<lower=0> sigmac2;             // case rate
real<lower=0> sigmar;             // recovery rate
real<lower=0> sigmad0;             // death rate
real<lower=0> sigmad1;
real<lower=0> sigmad2;
real<lower=0> alpha;
}

transformed parameters {

  real dI[n_weeks];
  real dC[n_weeks];
  real dR[n_weeks];
  real dD[n_weeks];
  real sigmac[n_weeks];
  real sigmad[n_weeks];
{
  real C;
  real I;
  real Cd[n_weeks];

  C = 0;
  I = 0;
  for (i in 1:n_weeks){
    sigmac[i] = sigmac0 + sigmac1*i^2/(1+sigmac2*i^2);
    sigmad[i] = sigmad0+ sigmad1/(1 + sigmad2*i^2);
    I += alpha;
    I *= exp(beta[i] - sigmac[i] - sigmau);
    dC[i] = sigmac[i]*I;
    C += dC[i];
    C *= exp(-(sigmar+sigmad[i]));
    Cd[i] = C;
    dI[i] = (exp(beta[i])-1)*I + alpha;
    dR[i] = sigmar*C;
    if (i > 2)
      dD[i] = sigmad[i]*Cd[i-2];
    else
      dD[i] = sigmad[i]*C;
  }
  }
}

model {
  sigmac0 ~ exponential(.25);
  sigmac1 ~ exponential(.25);
  sigmac2 ~ normal(15.,1.);
  sigmau ~ exponential(5.);
  sigmar ~ exponential(4.);
  sigmad0 ~ exponential(1.);
  sigmac1 ~ exponential(1.);
  sigmac2 ~ normal(15.,1.);
  alpha ~ exponential(10.);

    for (i in 1:n_weeks){
      //lambda[i] ~ cauchy(0.,1.);
      beta[i] ~ exponential(.5);
      target += poisson_lpmf(y_wk[i,1] | dC[i]);
      target += poisson_lpmf(y_wk[i,2] | dR[i]);
      target += poisson_lpmf(y_wk[i,3] | dD[i]);
    }
    for (i in 2:n_weeks-1){
      target += normal_lpdf(beta[i+1]-beta[i] | 0, .02);
      target += normal_lpdf(beta[i+1]-2*beta[i]+beta[i-1] | 0, .3);
    }
}

// generated quantities

generated quantities {

    real car[n_weeks];
    real ifr[n_weeks];
    real Rt[n_weeks];
    int y_proj[n_weeks*7,n_ostates];

    {
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
      Rt[i] = beta[i]/sigmac[i];
      for (j in 1:7){
        y_proj[7*(i-1) + j,1] = poisson_rng(min([dC[i]/7,1e8]));
        y_proj[7*(i-1) + j,2] = poisson_rng(min([dR[i]/7,1e8]));
        y_proj[7*(i-1) + j,3] = poisson_rng(min([dD[i]/7,1e8]));
      }
    }
    }
}
