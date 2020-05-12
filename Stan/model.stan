// model block for all SICR models
model {
    //priors Stan convention:  gamma(shape,rate), inversegamma(shape,rate)
    f1 ~ gamma(2.,1./10.);                 // f1  initital infected to case ratio
    f2 ~ gamma(1.5,1.);                    // f2  beta - sigmau
    sigmar ~ inv_gamma(4.,.2);             // sigmar
    sigmad ~ inv_gamma(2.78,.185);         // sigmad
    sigmau ~ inv_gamma(2.3,.15);           // sigmau
    extra_std ~ exponential(1.);           // likelihood over dispersion std
    q ~ exponential(1.);                   // q
    mbase ~ exponential(1.);               // mbase
    mlocation ~ lognormal(log(tm+5),1.);   // mlocation
    cbase ~ exponential(.2);               // cbase
    clocation ~ lognormal(log(20.),2.);    // clocation
    n_pop ~ lognormal(log(1e5),4.);        // population
    sigmar1 ~ inv_gamma(4.,.2);            // sigmar1

    // Likelihood function

    for (i in 1:n_obs){
      for (j in 1:3) {
        if (y[i,j] > 0.)
          target += neg_binomial_2_lpmf(y[i,j]|lambda[i,j],phi);
        }
    }
}
