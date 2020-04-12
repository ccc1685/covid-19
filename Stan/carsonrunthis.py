import pystan, pickle, sys
import pandas as pd
import numpy as np

roi = sys.argv[1]
#expects data to be in local folder called data
fname = "../data/covid_timeseries_"+roi+".csv"

n_chains=1
n_warmups=1000
n_iter=1100
n_thin=1
control = {'adapt_delta':0.98}

#modify lines 245-253 for compiling, saving, loading stan model

stancode = """
functions {
            real[] SIR(real t,  // time
            real[] u,           // system state {infected,cases,susceptible}
            real[] theta,       // parameters
            real[] x_r,
            int[] x_i) {

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

            real I = u[1];  // infected, latent
            real C = u[2];  // cases, observed

            real m;
            real c;

            m = mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)));  // mitigation

            c = 1 + cmax*t/(c50+t);    // case detection time dependence

            beta *= m;
            sigmac *= c;

            du_dt[1] = beta*(I+q*C) - sigmac*I - sigmau*I; //I
            du_dt[2] = sigmac*I - (sigmar+sigmad)*C;       //C
            du_dt[3] = beta*(I+q*C);                       //N_I
            du_dt[4] = sigmac*I; // N_C case appearance rate
            du_dt[5] = sigmar*C; // R_C appearance rate
            du_dt[6] = sigmad*C; // D_C appearance rate

            return du_dt;
          }


        }

        data {
          int<lower = 1> n_obs;       // number of days observed
          int<lower = 1> n_theta;     // number of model parameters
          int<lower = 1> n_difeq;     // number of differential equations for yhat
          int<lower = 1> n_ostates;   // number of observed states
          real<lower = 1> n_scale;    // scale to match observed scale
          int y[n_obs,n_ostates];     // data, per-day-tally [cases,recovered,death]
          real t0;                    // initial time point
          real tm;                    // start day of mitigation
          real ts[n_obs];             // time points that were observed
          }

        transformed data {
            real x_r[0];
            int x_i[0];
        }

        parameters {
            real<lower = 0> theta[n_theta]; // model parameters
        }

        transformed parameters{
            real u[n_obs, n_difeq];   // solution from the ODE solver
            real u_init[n_difeq];     // initial conditions for fractions

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
            real theta_init = theta[12];

            real lambda[n_obs,3]; //poisson parameter [cases, deaths, recovered]

            // What ODE system is expecting:
            // beta -> beta*m(t), sigmac -> sigmac*c(t)
            //du_dt[1] = beta*(I+q*C) - sigmac*c(t)*I - sigmau*I; //I
            //du_dt[2] = sigmac*c(t)*I - (sigmar+sigmad)*C; //C
            //du_dt[3] = beta*(I+q*C); //N_I
            //du_dt[4] = sigmac*I; //N_C
            //du_dt[5] = sigmar*C;// sigmarC
            //du_dt[6] = sigmad*C; // sigmadD

            u_init[1] = theta_init/n_scale; // I fitted
            u_init[2] = y[1,1]/n_scale; //C  from data
            u_init[3] = u_init[1];   // N_I cumulative infected
            u_init[4] = u_init[2] ;  // N_C cumulative cases
            u_init[5] = 0;  //sigmar C, case recovered rate
            u_init[6] = 0;  //sigmad D, case death rate

            //print(theta)
            u = integrate_ode_rk45(SIR, u_init, t0, ts, theta, x_r, x_i);

            //lambda
            lambda[1,1] = (u[1,4]-u_init[4])*n_scale; //C: cases sigma_c*\int I
            lambda[1,2] = (u[1,5]-u_init[5])*n_scale; //R: recovered sigma_r*\int C
            lambda[1,3] = (u[1,6]-u_init[6])*n_scale; //D: dead sigma_d * \int C

            for (i in 2:n_obs){
                lambda[i,1] = (u[i,4]-u[i-1,4])*n_scale; //C: cases sigma_c*\int_{interval} I
                lambda[i,2] = (u[i,5]-u[i-1,5])*n_scale; //R: recovered sigma_r*C
                lambda[i,3] = (u[i,6]-u[i-1,6])*n_scale; //D: dead

            }
        }

        model {

            //priors

            theta[1] ~ lognormal(log(0.3),3); //beta
            theta[2] ~ lognormal(log(0.2),3); //sigmac
            theta[3] ~ lognormal(log(0.1),1); //sigmar
            theta[4] ~ lognormal(log(0.1),1); //sigmad
            theta[5] ~ lognormal(log(0.01),3); //q
            theta[6] ~ lognormal(log(0.1),1); //sigmau
            theta[7] ~ lognormal(log(0.1),3); //mbase
            theta[8] ~ lognormal(log(tm),1); //mlocation
            theta[9] ~ lognormal(log(1),1.5); //mrate
            theta[10] ~ lognormal(log(0.1),3);//cmax
            theta[11] ~ lognormal(log(10),3);//c50
            theta[12] ~ lognormal(log(1),3);// theta_init

            //likelihood
            //lambda[1,1] =  sigma_c * I for day
            //lambda[1,2] =  sigma_r * C for day
            //lambda[1,3] =  sigma_d * C for day

            target += poisson_lpmf(y[1,1]|max([lambda[1,1],1.0])); //C
            target += poisson_lpmf(y[1,2]|max([lambda[1,2],1.0])); //R
            target += poisson_lpmf(y[1,3]|max([lambda[1,3],1.0])); //D

            for (i in 2:n_obs){
                target += poisson_lpmf(y[i,1]|max([lambda[i,1],1.0])); //C
                target += poisson_lpmf(y[i,2]|max([lambda[i,2],1.0])); //R
                target += poisson_lpmf(y[i,3]|max([lambda[i,3],1.0])); //D
            }
        }

        generated quantities {
            real R0;
            real car[n_obs];
            real attack_ratio[n_obs];
            real ifr[n_obs];

            //u[3]:NI
            //u[4]:NC
            //u[5]:NR
            //u[6]:ND

            real ll_; // log-likelihood for model

            //likelihood
            R0 = beta*(sigmar+sigmad+q*sigmac)/((sigmar+sigmad)*(sigmac+sigmau));

            car[1] = u[1,4]/u[1,3];
            attack_ratio[1] = u[1,3]; // u's are already scaled by n_scale
            ifr[1] = u[1,6]/u[1,3];

            ll_ = poisson_lpmf(y[1,1]|max([lambda[1,1],1.0]));
            ll_ += poisson_lpmf(y[1,2]|max([lambda[1,2],1.0]));
            ll_ += poisson_lpmf(y[1,3]|max([lambda[1,3],1.0]));


            for (i in 2:n_obs){
                car[i] = u[i,4]/u[i,3];
                attack_ratio[i] = u[i,3]; // u's are already scaled by n_scale
                ifr[i] = u[i,6]/u[i,3];

                ll_ += poisson_lpmf(y[i,1]|max([lambda[i,1],1.0]));
                ll_ += poisson_lpmf(y[i,2]|max([lambda[i,2],1.0]));
                ll_ += poisson_lpmf(y[i,3]|max([lambda[i,3],1.0]));
            }

            print(ll_)

        }
        """

##uncomment to compile model
# stanrunmodel = pystan.StanModel(model_code=stancode)

##uncomment to save compiled model
# with open('stanrunmodel.pkl', 'wb') as f:
#    pickle.dump(stanrunmodel, f)

##uncomment to load compiled model
stanrunmodel = pickle.load(open('stanrunmodel.pkl', 'rb'))

# # Load data from Gerkin scrape

DF = pd.read_csv(fname)

# t0 := where to start time series, index space
t0 = np.where(DF["new_cases"].values>=10)[0][0]
# tm := start of mitigation, index space
tm = t0 + 10


# ## Format JHU ROI data for Stan

# int<lower = 1> n_obs;       // number of days observed
# int<lower = 1> n_theta;     // number of model parameters
# int<lower = 1> n_difeq;     // number of differential equations for yhat
# int<lower = 1> n_ostates;     // number of observed states
# int<lower = 1> n_pop;       // population
# real<lower = 1> n_scale;       // scale to match observed scale
# int y[n_obs,n_ostates];           // data, per-day-tally [cases,recovered,death]
# real t0;                // initial time point
# real tm; //start day of mitigation
# real ts[n_obs];         // time points that were observed
# int<lower = 1> n_obs_predict;       // number of days to predict
# real ts_predict[n_obs_predict];         //

stan_data = {}

# stan_data['n_pop'] = pop[roi]
stan_data['n_scale'] = 100000#10000000 #use this instead of population

stan_data['n_theta'] = 12
stan_data['n_difeq'] = 6
stan_data['n_ostates'] = 3

stan_data['t0'] = t0-1 #to for ODE is one day, index before start of series
stan_data['tm'] = tm
stan_data['ts'] = np.arange(t0,len(DF['dates2']))
# DF = DF.replace('NaN', 0)
stan_data['y'] = (DF[['new_cases','new_recover','new_deaths']].to_numpy()).astype(int)[t0:,:]
# stan_data['y'][stan_data['y']<0] = 0
stan_data['n_obs'] = len(DF['dates2']) - t0


# theta[1] ~ lognormal(log(0.25),1); //beta
# theta[2] ~ lognormal(log(0.1),1); //sigmac
# theta[3] ~ lognormal(log(0.01),1); //sigmar
# theta[4] ~ lognormal(log(0.01),1); //sigmad
# theta[5] ~ lognormal(log(0.01),1); //q
# theta[6] ~ lognormal(log(1),1); //sigmau
# theta[7] ~ lognormal(log(0.1),1); //mbase
# theta[8] ~ lognormal(log(tm),5); //mlocation
# theta[9] ~ lognormal(log(1),5); //mrate
# theta[10] ~ lognormal(log(0.1),1);//cmax
# theta[11] ~ lognormal(log(10),1);//c50
# theta[12] ~ lognormal(log(1),1);// theta_init
#init = [{'theta':[0.25,0.1,0.01,0.01,0.01,1.0,0.1,1.0,1.0,0.1,10.0,1.0]}]

## function used to initialize parameters
def init_fun():
        x = {'theta':
             [np.random.lognormal(np.log(0.3),2)]+
             [np.random.lognormal(np.log(0.2),2)]+
             [np.random.lognormal(np.log(0.1),2)]+
             [np.random.lognormal(np.log(0.1),2)]+
             [np.random.lognormal(np.log(0.01),2)]+
             [np.random.lognormal(np.log(0.1),1)]+
             [np.random.lognormal(np.log(0.1),1)]+
             [np.random.lognormal(np.log(stan_data['tm']),3)]+
             [np.random.lognormal(np.log(1),3)]+
             [np.random.lognormal(np.log(0.1),1)]+
             [np.random.lognormal(np.log(10),1)]+
             [np.random.lognormal(np.log(1),.2)]
            }
        return x


# ## Fit Stan

fit = stanrunmodel.sampling(data = stan_data,init = init_fun ,control=control, chains = n_chains,chain_id=np.arange(n_chains), warmup = n_warmups, iter = n_iter, thin=n_thin, seed=13219)

## save fit
import pickle
with open("model_fit_"+roi+".pkl", "wb") as f:
    pickle.dump({'model' : stanrunmodel, 'fit' : fit}, f, protocol=-1)

## uncomment to print fit summary
print(fit)
