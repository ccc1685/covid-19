generated quantities {
    real ll_; // log-likelihood for model
    real llx[n_obs, 3];

    ll_ = 0;
    for (i in 1:n_obs) {
        for (j in 1:3) {
            llx[i, j] = neg_binomial_2_lpmf(max(y[i,j],0)|max([lambda[i,j],1.0]),phi);
            ll_ += llx[i, j];
            }
        }
    }
