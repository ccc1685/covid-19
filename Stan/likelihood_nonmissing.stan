real


if(y[1,1]>0){
target += neg_binomial_2_lpmf(y[1,1]|lambda[1,1]),phi); //C
}
if(y[1,2]>0){
target += neg_binomial_2_lpmf(y[1,2]|lambda[1,2],phi); //R
}
if(y[1,3]>0){
target += neg_binomial_2_lpmf(y[1,3]|lambda[1,3],phi); //D
}

for (i in 2:n_obs){
    if(y[i,1]>0){
    target += neg_binomial_2_lpmf(y[i,1]|lambda[i,1],phi); //C
    }
    if(y[i,2]>0){
    target += neg_binomial_2_lpmf(y[i,2]|lambda[i,2],phi); //R
    }
    if(y[i,3]>0){
    target += neg_binomial_2_lpmf(y[i,3]|lambda[i,3],phi); //D
    }
}
