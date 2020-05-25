// generated quantities

generated quantities {
    real ll_; // log-likelihood for model
    real llx[n_obs, 3];

    ll_ = 0;
    for (i in 1:n_obs) {
        for (j in 1:3) {
           if (y[i,j] > 0.)
               llx[i, j] = neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi);
           else
               llx[i,j] = 0.;
            ll_ += llx[i, j];
            }
        }
    }
