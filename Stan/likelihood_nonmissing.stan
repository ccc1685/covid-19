if(y[1,1]>0){
target += neg_binomial_2_lpmf(y[1,1]|max([lambda[1,1],1.0]),phi); //C
}
if(y[1,2]>0){
target += neg_binomial_2_lpmf(y[1,2]|max([lambda[1,2],1.0]),phi); //R
}
if(y[1,3]>0){
target += neg_binomial_2_lpmf(y[1,3]|max([lambda[1,3],1.0]),phi); //D
}

for (i in 2:n_obs){
    if(y[i,1]>0){
    target += neg_binomial_2_lpmf(y[i,1]|max([lambda[i,1],1.0]),phi); //C
    }
    if(y[i,2]>0){
    target += neg_binomial_2_lpmf(y[i,2]|max([lambda[i,2],1.0]),phi); //R
    }
    if(y[i,3]>0){
    target += neg_binomial_2_lpmf(y[i,3]|max([lambda[i,3],1.0]),phi); //D
    }
}
