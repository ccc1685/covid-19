    functions {
    real[] SIR(
            real t,  // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters 
            real[] x_r,
            int[] x_i) 
        {
            real du_dt[6];
        
            real beta = theta[1];
            real sigmac = theta[2];
            real sigmar = theta[3];
            real sigmad =  theta[4];
            real q = theta[5]; 
            real sigmau = theta[6]; 
            real mbase = theta[7]; 
            real mlocation = theta[8]; 
            real mrate = theta[9]; 
            real cmax = theta[10];
            real c50 = theta[11];
            
            real I = u[1];  # unknown infected
            real C = u[2];  # cases
            
            real m;
            real c;
            
            m = mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)));            
            
            c = 1 + cmax*t/(c50+t);
            
            beta *= m;
            sigmac *= c;
            
            du_dt[1] = beta*(I+q*C) - sigmac*I - sigmau*I; //I
            du_dt[2] = sigmac*I - (sigmar+sigmad)*C; //C
            du_dt[3] = beta*(I+q*C); //N_I
            du_dt[4] = sigmac*I; //N_C case appearance rate
            du_dt[5] = sigmar*C; // R_C appearance rate
            du_dt[6] = sigmad*C; // D_C appearance rate
            
            return du_dt;
        }
}

data {
    int<lower = 1> N;               // number of observationss
    int<lower = 1> R;               // number of regions
    int<lower = 1> N_regional[R];   // number of observations per region

    real<lower = 1> n_scale[R];     // scale to match observed scale
    int cases[N];                   // daily new cases
    int deaths[N];                  // daily new deaths
    int recovered[N];               // daily new recovered
    real t[N];                      // observed times
    real tm[R];                     // regional start day of mitigation
}

transformed data {
    real x_r[0];
    int x_i[0];
    real times[70] = {
      1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
      47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
      69, 70
      };
}

parameters {
    real<lower=0> beta[R];
    real<lower=0> beta_mu;
    real<lower=0> beta_sigma;

    real<lower=0> sigmac[R];
    real<lower=0> sigmac_mu;
    real<lower=0> sigmac_sigma;

    real<lower=0> sigmar[R];
    real<lower=0> sigmar_mu;
    real<lower=0> sigmar_sigma;

    real<lower=0> sigmad[R];
    real<lower=0> sigmad_mu;
    real<lower=0> sigmad_sigma;

    real<lower=0> q[R];
    real<lower=0> q_mu;
    real<lower=0> q_sigma;

    real<lower=0> sigmau[R];
    real<lower=0> sigmau_mu;
    real<lower=0> sigmau_sigma;

    real<lower=0> mbase[R];
    real<lower=0> mbase_mu;
    real<lower=0> mbase_sigma;

    real<lower=0> mlocation[R];
    real<lower=0> mlocation_mu;
    real<lower=0> mlocation_sigma;

    real<lower=0> mrate[R]; 
    real<lower=0> mrate_mu;
    real<lower=0> mrate_sigma;

    real<lower=0> cmax[R];
    real<lower=0> cmax_mu;
    real<lower=0> cmax_sigma;

    real<lower=0> c50[R];
    real<lower=0> c50_mu;
    real<lower=0> c50_sigma;

    real<lower=0> I0[R];
    real<lower=0> C0[R];
}

transformed parameters{

    // The lambda parameters are sufficient for determining the likelihood
    real lambda_cases[N];
    real lambda_deaths[N];
    real lambda_recovered[N];
    {
        int pos = 1;

        for (region in 1:R){
            int N_region = N_regional[region];
            real u_init[6];                     // initial conditions for fractions
            real u[N_region, 6];                // solution from the ODE solver
            real theta[11] = {
                beta[region], sigmac[region], sigmar[region], sigmad[region],
                q[region], sigmau[region], mbase[region], mlocation[region],
                mrate[region], cmax[region], c50[region]};

            int regional_deaths[N_region] = segment(deaths, pos, N_region);
            int regional_cases[N_region] = segment(cases, pos, N_region);
            int regional_recovered[N_region] = segment(recovered, pos, N_region);

            u_init[1] = I0[region]/n_scale[region]; // I
            u_init[2] = C0[region]/n_scale[region]; //C
            u_init[3] = u_init[1];  //N_I
            u_init[4] = u_init[2] ;  //N_C
            u_init[5] = 0;  //sigmarC
            u_init[6] = 0;  //sigmadD 

            u = integrate_ode_rk45(SIR, u_init, (t[pos]-1), times, theta, x_r, x_i);

            lambda_cases[pos] = (u[1, 4] - u_init[4])*n_scale[region];
            lambda_deaths[pos] = (u[1, 6] - u_init[6])*n_scale[region];
            lambda_recovered[pos] = (u[1, 5] - u_init[5])*n_scale[region];

            for (n in 2:N_region){
                lambda_cases[pos+n-1] = (u[n, 4] - u[n-1, 4])*n_scale[region];
                lambda_deaths[pos+n-1] = (u[n, 6] - u[n-1, 6])*n_scale[region];
                lambda_recovered[pos+n-1] = (u[n, 5] - u[n-1, 5])*n_scale[region];
            }

            pos += N_region;
        }
    }

}

model {

    beta ~ lognormal(beta_mu, beta_sigma);              //beta 
    beta_mu ~ lognormal(log(0.25), 0.5);
    beta_sigma ~ cauchy(0, 0.5);

    sigmac ~ lognormal(sigmac_mu, sigmac_sigma);             //sigmac
    sigmac_mu ~ lognormal(log(0.1),1);
    sigmac_sigma ~ cauchy(0, 0.5);

    sigmar ~ lognormal(sigmar_mu, sigmar_sigma);            //sigmar
    sigmar_mu ~ lognormal(log(0.01),1);
    sigmar_sigma ~ cauchy(0, 0.5);

    sigmad ~ lognormal(sigmad_mu, sigmad_sigma);            //sigmad
    sigmad_mu ~ lognormal(log(0.01),1); 
    sigmad_sigma ~ cauchy(0, 0.5);

    q ~ lognormal(q_mu, q_sigma);                 //q
    q_mu ~ lognormal(log(0.01),1); 
    q_sigma ~ cauchy(0, 0.5);

    sigmau ~ lognormal(sigmau_mu, sigmau_sigma);            //sigmau
    sigmau_mu ~ lognormal(log(0.01),1);
    sigmau_sigma ~ cauchy(0, 0.5);

    mbase ~ lognormal(log(0.1),1);

    mlocation ~ lognormal(log(tm),5);           //mlocation 

    mrate ~ lognormal(log(1),5);                //mrate

    cmax ~ lognormal(log(0.1),1);               //cmax 
    c50 ~ lognormal(log(10),1);                 //c50 

    for(n in 1:N){
        target += poisson_lpmf(cases[n]|max([lambda_cases[n],0.0]));
        target += poisson_lpmf(recovered[n]|max([lambda_recovered[n],0.0]));
        target += poisson_lpmf(deaths[n]|max([lambda_deaths[n],0.0]));
    }

}

generated quantities {
/* Skip this for now
    real ll_; // log-likelihood for model
    
    //likelihood
    ll_ = poisson_lpmf(y[1,1]|max([lambda[1,1],0.0]));
    ll_ += poisson_lpmf(y[1,2]|max([lambda[1,2],0.0]));
    ll_ += poisson_lpmf(y[1,3]|max([lambda[1,3],0.0]));

    
    for (i in 2:n_obs){
        ll_ += poisson_lpmf(y[i,1]|max([lambda[i,1],0.0]));
        ll_ += poisson_lpmf(y[i,2]|max([lambda[i,2],0.0]));
        ll_ += poisson_lpmf(y[i,3]|max([lambda[i,3],0.0]));
    }
    
    print(ll_)
    */
    
}
