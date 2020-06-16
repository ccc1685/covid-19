// Likelihood function

for (i in 1:n_obs){
  for (j in 1:3) {
    if (y[i,j] > -1.)
      target += neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi);
    }
}
