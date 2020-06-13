// generated quantities

generated quantities {
    real llx[n_obs, 3];
    real ll_; // log-likelihood for model
    int n_data_pts;
    real y_proj[n_total,3];

    ll_ = 0;
    n_data_pts = 0;
    for (i in 1:n_obs) {
        for (j in 1:3) {
           if (y[i,j] > 0.){
                llx[i, j] = neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi);
                n_data_pts += 1;
                ll_ += llx[i, j];
               }
           else {
                llx[i,j] = 0.;
               }
          }
    }

    for (i in 1:n_total) {
        for (j in 1:3) {
            y_proj[i,j] = poisson_rng(lambda[i,j]);
        }
    }
}
