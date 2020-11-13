// SICR2HV.stan
// Latent variable nonlinear SICR model with mitigation and release, q>0

functions {

    // nonlinear SICR model with R delay ODE function
       real[] SICR(
       real t,             // time
       real[] u,           // system state {infected,cases,susceptible}
       real[] theta,       // parameters
       real[] x_r,
       int[] x_i
       )
       {
         real du_dt[11];
         real f1 = theta[1];          // beta - sigmau - sigmac
         real f2 = theta[2];          // beta - sigma u
         real sigmaVS = theta[3];
         real sigmaSR = theta[4];
         real sigmaSRI = theta[5];
         real sigmaIV = theta[6];
         real sigmaRI = theta[7];
         real sigmaDI = theta[8];
         real sigmaRC = theta[9];
         real sigmaDC = theta[10];
         real sigmaRH1 = theta[11];
         real sigmaRH2 = theta[12];
         real sigmaDH2 = theta[13];
         real sigmaRV = theta[14];
         real sigmaH1C = theta[15];
         real sigmaH1V = theta[16];
         real sigmaH2H1 = theta[17];
         real sigmaDH1 = theta[18];
         real q = theta[19];
         real sigmaM = theta[20];
         real mbase = theta[21];

         real S = u[1];  // susceptibles, latent
         real I = u[2];  // infected, latent
         real C = u[3];  // cases, observed
         real R = u[4];  // recovery compartment from C, observed
         real RI = u[5]; // recovery compartment from I, latent
         real H1 = u[6]; // hospital compartment 1, observed
         real H2 = u[7]; // hosplital compartment 2, observed
         real V = u[8];  // vaccinated
         real M = u[11]; // mitigation

         real sigmaCI = f2/(1+f1);
         real beta = f2 + sigmaRI + sigmaDI;
         real Cdot = sigmaCI*I - sigmaRC*C - sigmaDC*C - sigmaH1C*C;

         du_dt[1] = -beta*(I+q*C)*S*M - sigmaVS*S + sigmaSR*R + sigmaSRI*RI ; //dS/dt
         du_dt[2] = beta*(I+q*C)*S*M + sigmaIV*V - sigmaCI*I - sigmaRI*I - sigmaDI*I;     //dI/dt
         du_dt[3] = Cdot;        //dC/dt
         du_dt[4] = sigmaRC*C + sigmaRH1*H1 + sigmaRH2*H2 + sigmaRV*V - sigmaSR*R;     // dR/dt
         du_dt[5] = sigmaRI*I - sigmaSRI*RI;                               // dRI/dt
         du_dt[6] = sigmaH1C*C + sigmaH1V*V - sigmaH2H1*H1 - sigmaRH1*H1 - sigmaDH1*H1; // dH1/dt
         du_dt[7] = sigmaH2H1*H1 - sigmaRH2*H2 - sigmaDH2*H2;             // dH2/dt
         du_dt[8] = sigmaVS*S - sigmaIV*V - sigmaRV*V - sigmaH1V*V;   // dV/dt
         du_dt[9] = sigmaVS*S;                                              // cum V
         du_dt[10] = Cdot - u[10]/14;     // 14 day moving average of change in C
         du_dt[11] = sigmaM*(1 - M) - mbase*max([u[10],0])*M;  // mitigation

         return du_dt;

       }
  }


data {
      int<lower=1> n_obs;       // number of days observed
      int n_total;               // total number of days simulated, n_total-n_obs is days beyond last data point
      int<lower=1> n_ostates;   // number of observed states
      int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
      real tm;                    // start day of mitigation
      real ts[n_total];             // time points that were observed + projected
  }


transformed data {
    real x_r[0];
    int x_i[0];
    real sigmaVS = 0;
    real sigmaIV = 0;
    real sigmaH1V = 0;
    real sigmaRV = 0;
    real sigmaSR = 0;
    real sigmaSRI = 0;

}

parameters {

    real<lower=0> f1;             //  sigmaCI = f2/(1+f1), initial infected to case ratio
    real<lower=0> f2;             // f2  beta - sigmaCI - sigmaDI
    //real<lower=0> sigmaVS;
    //real<lower=0> sigmaSR;
    //real<lower=0> sigmaSRI;

    real<lower=0> sigmaRI;
    real<lower=0> sigmaDI;
    real<lower=0> sigmaRC;
    real<lower=0> sigmaDC;
    real<lower=0> sigmaH1C;
    real<lower=0> sigmaRH1;
    real<lower=0> sigmaDH1;
    real<lower=0> sigmaH2H1;
    real<lower=0> sigmaRH2;
    real<lower=0> sigmaDH2;
    //real<lower=0> sigmaRV;
    //real<lower=0> sigmaIV;
    //real<lower=0> sigmaH1V;

    real<lower=0> sigmaM;
    real<lower=0> mbase;
    real<lower=0> q;              // infection factor for cases
    real<lower=1> n_pop;      // population size


}

transformed parameters{

  real sigmaCI = f2/(1+f1);
  real beta = f2 + sigmaRI + sigmaDI;
  real lambda[n_total,5];     // Poisson rate [new cases, new recovered, new deaths, new hosp, new ICU]
  //real frac_infectious[n_total];
  //real cumulativeV[n_total];


  {
     real theta[21] = {f1,f2,sigmaVS,sigmaSR,sigmaSRI,sigmaIV,sigmaRI,sigmaDI,sigmaRC,sigmaDC,sigmaRH1,sigmaRH2,sigmaDH2,sigmaRV,sigmaH1C,sigmaH1V,sigmaH2H1,sigmaDH1,q,sigmaM,mbase};
     real u[n_total, 11];   // solution from the ODE solver
     real cinit = y[1,1]/n_pop;
     real u_init[11];     // initial conditions for fractions

     u_init[1] = 1;                 //S
     u_init[2] = f1*cinit;         // I, f1 * Cinit
     u_init[3] = cinit;            // C, from data
     u_init[4] = 0;             // R
     u_init[5] = 0;             // RI
     u_init[6] = 0;             // H1
     u_init[7] = 0;             // H2
     u_init[8] = 0;             // V
     u_init[9] = 0;             // cum V
     u_init[10] = 0;
     u_init[11] = 1;            // M

     u = integrate_ode_rk45(SICR, u_init, ts[1]-1, ts, theta, x_r, x_i,1e-3,1e-5,2000);

     //print(u_init);

     for (i in 1:n_total){

        lambda[i,1] = sigmaCI*u[i,2]*n_pop;  //C: cases per day
        lambda[i,2] = (sigmaRC*u[i,3] + sigmaRH1*u[i,6] + sigmaRH2*u[i,7] +sigmaRV*u[i,8])*n_pop; // new R per day
        lambda[i,3] = (sigmaDC*u[i,3] + sigmaDH1*u[i,6] + sigmaDH2*u[i,7])*n_pop; //new D per day
        lambda[i,4] = (sigmaH1C*u[i,3] + sigmaH1V*u[i,8])*n_pop; //new H1 per day
        lambda[i,5] = sigmaH2H1*u[i,6]*n_pop;                 //new H2 per day
        //lambda[i,6] = sigmaVS*u[i,1]; //new V per day
        //frac_infectious[i] = u[i,2];
        //cumulativeV[i] = u[i,9]*n_pop;
        }

    }
}


model {
//priors Stan convention:  exponential(rate), gamma(shape,rate), inversegamma(shape,rate)

    f1 ~ normal(1.75,.1);                     // f1  initital infected to case ratio
    f2 ~ normal(.25,.1);                 // f2  beta - sigmau

    //sigmaVS ~ exponential(7.);
    //sigmaSR ~ exponential(360.);
    //sigmaSRI ~ exponential(360.);
    //sigmaIV ~ exponential(100.);
    sigmaRI ~ normal(.01,.003);
    sigmaDI ~ normal(.03,.01);
    sigmaRC ~ normal(.08,.03);
    sigmaDC ~ normal(.003,.001);
    sigmaRH1 ~ normal(1.,.3);
    sigmaRH2 ~ normal(.1,.03);
    //sigmaRV ~ normal(31.);
    sigmaH1C ~ normal(.01,.003);
    //sigmaH1V ~ normal(30.);
    sigmaH2H1 ~ normal(.4,.1);
    sigmaDH1 ~ normal(.003,.001);
    sigmaDH2 ~ normal(.001,.0003);
    sigmaM ~ exponential(1.);
    mbase ~ exponential(1.);
    q ~ exponential(2.);                   // q
    n_pop ~ normal(4e6,1e6);        // population


// Likelihood function

for (i in 1:n_obs){
  for (j in 1:n_ostates) {
    if (y[i,j] > -1.)
      target += poisson_lpmf(y[i,j]|max([lambda[i,j],.01]));
    }
}

}

// generated quantities

generated quantities {
  //  real llx[n_obs, n_ostates];
  //  real ll_; // log-likelihood for model
  //  int n_data_pts;
    real y_proj[n_total,n_ostates];
/*
    ll_ = 0;
    n_data_pts = 0;
    for (i in 1:n_obs) {
        for (j in 1:n_ostates) {
           if (y[i,j] > -1.){
                llx[i, j] = poisson_lpmf(y[i,j]|max([lambda[i,j],.01]));
                n_data_pts += 1;
                ll_ += llx[i, j];
               }
           else {
                llx[i,j] = 0.;
               }
          }
    }
*/
    for (i in 1:n_total) {
        for (j in 1:n_ostates) {
          if (lambda[i,j] < 1e8)
            y_proj[i,j] = poisson_rng(max([lambda[i,j],.01]));
          else
            y_proj[i,j] = poisson_rng(1e8);
        }
    }

}
