Julia file SIRmcmc.jl was used to repeat the SIR analysis of Lourenco et al. (2020). SIRmcmc.jl contains the function mcmc(data,r,total) to run a Metropolis-Hastings MCMC on the model parameters, where data is the cumulative death time series starting from the first death, r is the set of 4 SIR ODE model parameters, and total is the number of MCMC samples. The SIR ODE model is implemented in two versions of the function oxford(zinit,yinit,total,dt,beta,sigma,rtheta,psi), which use different numerical integration methods. The function mcmc outputs the sample series for the loglikelihood and parameters as well as the maximum likelihood parameters. The parameter guess at each step is implemented in sampleparams!(rt,r), which uses a lognormal guess distribution. The loglikelihood is computed in the function loss(data,r) which uses a Poisson likelihood function with rate given by the predicted death number from the model. The model can be called to directly fit all four parameters with model(r) or three parameters with R0 constrained to 2.25 with modelconstrained(r).

Other Julia files contain code to run Metropolis-Hastings MCMC on various latent variable SIR models with varying degrees of complexity. These were used to prototype the models that were eventually fit using NUTS in Stan.

Required Julia packages:
```"Arpack", "CSV", "DataFrames", "DelimitedFiles", "DifferentialEquations", "Distributed", "Distributions", "NLsolve", "PyPlot", "StochasticDiffEq"
```

Example installation
```
julia -e 'Pkg.init()' && \
julia -e 'Pkg.update()' && \
    julia -e 'Pkg.add("Arpack", "CSV", "DataFrames", "DelimitedFiles", "DifferentialEquations", "Distributed", "Distributions", "NLsolve", "PyPlot", "StochasticDiffEq")' && \
    # Precompile Julia packages \
    julia -e 'Pkg.build("Arpack", "CSV", "DataFrames", "DelimitedFiles", "DifferentialEquations", "Distributed", "Distributions", "NLsolve", "PyPlot", "StochasticDiffEq")'
```

Example entrypoint
```
julia -p 8 SICRPoissonscriptBW.jl
```
