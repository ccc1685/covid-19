// SICRdiscrete.stan
// Discrete SICR model


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
      if (y_wk[i,k] < 0 )
        y_wk[i,k] = 0;
    }
  }
}


parameters {
real<lower=0> beta[n_weeks];             // infection rate
real<lower=0> sigmau;             // uninfected rate
real<lower=0> sigmac0;             // case rate
real<lower=0> sigmac1;             // case rate
real<lower=0> sigmar0;             // recovery rate
real<lower=0> sigmar1;             // recovery rate
real<lower=0> sigmad0;             // death rate
real<lower=0> sigmad1;
real<lower=0> alpha[n_weeks];
}

transformed parameters {

  real dI[n_weeks];
  real dC[n_weeks];
  real dR[n_weeks];
  real dD[n_weeks];
  real sigmac[n_weeks];
  real sigmad[n_weeks];
  real sigmar[n_weeks];

{
  real C;
  real I;
  real Cd[n_weeks];

  C = 0;
  I = 0;
  for (i in 1:n_weeks){
    sigmac[i] = sigmac0 + (sigmac1)*i;
    sigmad[i] = sigmad0/(1 + ((.1 + sigmad1)*i)^2);
    sigmar[i] = sigmar0/(1 + ((.1 + sigmar1)*i)^2);
    I += alpha[i];
    I *= exp(beta[i] - sigmac[i] - sigmau);
    dC[i] = sigmac[i]*I;
    C += dC[i];
    C *= exp(-(sigmar[i]+sigmad[i]));
    Cd[i] = C;
    dI[i] = (exp(beta[i])-1)*I + alpha[i];
    dR[i] = sigmar[i]*C;
    dD[i] = sigmad[i]*C;
    /*
    if (i > 2)
      dD[i] = sigmad[i]*Cd[i-2];
    else
      dD[i] = sigmad[i]*C;
      */
  }
  }
}

model {
  sigmac0 ~ exponential(.25);
  sigmac1 ~ exponential(.25);
  sigmau ~ exponential(5.);
  sigmar0 ~ exponential(1.);
  sigmar1 ~ exponential(1.);
  sigmad0 ~ exponential(1.);
  sigmad1 ~ exponential(1.);

    for (i in 1:n_weeks){
      alpha[i] ~ exponential(10.);
      beta[i] ~ exponential(.5);
      target += poisson_lpmf(y_wk[i,1] | dC[i]);
      target += poisson_lpmf(y_wk[i,2] | dR[i]);
      target += poisson_lpmf(y_wk[i,3] | dD[i]);
    }
    for (i in 2:n_weeks-1){
      target += normal_lpdf(beta[i+1]-beta[i] | 0, .2);
      target += normal_lpdf(beta[i+1]-2*beta[i]+beta[i-1] | 0, .5);
    }
}

// generated quantities

generated quantities {

    real car[n_weeks];
    real ifr[n_weeks];
    real Rt[n_weeks];
    int y_proj[n_weeks*7,n_ostates];
    real llx[n_weeks*7, 3];
    real ll_; // log-likelihood for model
    int n_data_pts;

    real R0 = beta[1]/sigmac[1];

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
          llx[7*(i-1) + j,1] = poisson_lpmf(y[7*(i-1) + j,1] | min([dC[i]/7,1e8]));
          llx[7*(i-1) + j,2] = poisson_lpmf(y[7*(i-1) + j,2] | min([dR[i]/7,1e8]));
          llx[7*(i-1) + j,3] = poisson_lpmf(y[7*(i-1) + j,3] | min([dD[i]/7,1e8]));
          ll_ += llx[7*(i-1) + j,1] + llx[7*(i-1) + j,2] + llx[7*(i-1) + j,3];
          n_data_pts += 1;
        }
      }
    }

}
