transformed parameters{
  real lambda[n_obs,3];     //neg_binomial_2 rate [new cases, new recovered, new deaths]
  real car[n_obs];          //total cases / total infected
  real ifr[n_obs];          //total dead / total infected
  real Rt[n_obs];           // time dependent reproduction number

  real u_init[7];     // initial conditions for fractions

  real sigmac = f2/(1+f1);
  real beta = f2 + sigmau;
  real sigma = sigmar + sigmad;
  real R0 = beta*(sigma+q*sigmac)/sigma/(sigmac+sigmau);   // reproduction number
  real phi = max([1/(extra_std^2),1e-10]); // likelihood over-dispersion of std

  {
     real theta[11] = {f1,f2,sigmar,sigmad,sigmau,q,mbase,mlocation,cbase,clocation,sigmar1};

     real u[n_obs, 7];   // solution from the ODE solver
     real sigmact;
     real betat;

     real cinit = y[1,1]/n_pop;
     u_init[1] = f1*cinit;      // I set from f1 * C initial
     u_init[2] = cinit;         // C from data
     u_init[3] = u_init[1];     // N_I cumulative infected
     u_init[4] = cinit;         // N_C total cumulative cases
     u_init[5] = cinit;         // total D
     u_init[6] = 0;             // R1
     u_init[7] = 0;             // total R

     u = integrate_ode_rk45(SICR, u_init, ts[1]-1, ts, theta, x_r, x_i,1e-3,1e-3,2000);

     for (i in 1:n_obs){
        car[i] = u[i,4]/u[i,3];
        ifr[i] = u[i,5]/u[i,3];
        betat = beta*transition(mbase,mlocation,i);
        sigmact = sigmac*transition(cbase,clocation,i);
        Rt[i] = betat*(sigma+q*sigmact)/sigma/(sigmact+sigmau);
        }

     lambda[1,1] = max([(u[1,4]-u_init[4])*n_pop,1.0]); //C: cases per day
     lambda[1,2] = max([(u[1,7]-u_init[7])*n_pop,1.0]); //R: recovered per day
     lambda[1,3] = max([(u[1,5]-u_init[5])*n_pop,1.0]); //D: dead per day

     for (i in 2:n_obs){
        lambda[i,1] = max([(u[i,4]-u[i-1,4])*n_pop,1.0]); //C: cases per day
        lambda[i,2] = max([(u[i,7]-u[i-1,7])*n_pop,1.0]); //R: recovered per day
        lambda[i,3] = max([(u[i,5]-u[i-1,5])*n_pop,1.0]); //D: dead per day
        }

    }
}
