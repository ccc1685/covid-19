target += neg_binomial_2_lpmf(max(y[1,1],0)|max([lambda[1,1],1.0]),phi); //C
target += neg_binomial_2_lpmf(max(y[1,2],0)|max([lambda[1,2],1.0]),phi); //R
target += neg_binomial_2_lpmf(max(y[1,3],0)|max([lambda[1,3],1.0]),phi); //D

for (i in 2:n_obs){
    target += neg_binomial_2_lpmf(max(y[i,1],0)|max([lambda[i,1],1.0]),phi); //C
    target += neg_binomial_2_lpmf(max(y[i,2],0)|max([lambda[i,2],1.0]),phi); //R
    target += neg_binomial_2_lpmf(max(y[i,3],0)|max([lambda[i,3],1.0]),phi); //D
}
