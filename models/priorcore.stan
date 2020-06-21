f1 ~ gamma(2.,.1);                 // f1  initital infected to case ratio
f2 ~ gamma(20,50.);                    // f2  beta - sigmau
sigmar ~ inv_gamma(4.,.2);             // sigmar
//sigmad ~ inv_gamma(2.78,.185);         // sigmad
sigmad ~ exponential(3.);
sigmau ~ inv_gamma(2.3,.15);           // sigmau
extra_std ~ exponential(2.);           // likelihood over dispersion std
