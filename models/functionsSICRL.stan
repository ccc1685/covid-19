
functions {
// time transition functions for beta and sigmac
#include transitionfn.stan

// linear SICR model
           real[] SICR(
           real t,             // time
           real[] u,           // system state {infected,cases,susceptible}
           real[] theta,       // parameters
           real[] x_r,
           int[] x_i
           )
           {
           real du_dt[5];
           real f1 = theta[1];          // beta - sigmau - sigmac
           real f2 = theta[2];          // beta - sigma u
           real sigmar = theta[3];
           real sigmad =  theta[4];
           real sigmau = theta[5];
           real q = theta[6];
           real mbase = theta[7];
           real mlocation = theta[8];
           real cbase = theta[9];
           real clocation = theta[10];

           real sigma = sigmar + sigmad;
           real sigmac = f2/(1+f1);
           real beta = f2 + sigmau;

           real I = u[1];  // infected, latent
           real C = u[2];  // cases, observed

           sigmac *= transition(cbase,clocation,t);  // case detection change
           beta *= transition(mbase,mlocation,t);  // mitigation

           du_dt[1] = beta*(I+q*C) - sigmac*I - sigmau*I; //I
           du_dt[2] = sigmac*I - sigma*C;       //C
           du_dt[3] = beta*(I+q*C);                       //N_I
           du_dt[4] = sigmac*I; // N_C case appearance rate
           du_dt[5] = C; // integrated C

           return du_dt;
         }
       }
