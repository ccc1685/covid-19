# Julia script to run set of models in tuple modeltypes on region
# User sets total number of mcmc steps, folders for input and output and temperature
# run using:  julia -p #nprocs SICRPoissonscrptBW.jl

t0 = time()

@everywhere include("SICR.jl")
@everywhere include("SICRmodels.jl")

datestrin = "2020-04-09linear"
datestrout = "2020-04-09linear"

total = Int(1e3)

precompile(runmcmcMP,(Int,String,String))
runmcmcMP(total,datestrin,datestrout)

println(" Time: ",time()-t0)
