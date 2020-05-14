data {
  int<lower=1> n_obs;       // number of days observed
  int<lower=1> n_ostates;   // number of observed states
  int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
  real tm;                    // start day of mitigation
  real ts[n_obs];             // time points that were observed
  }
