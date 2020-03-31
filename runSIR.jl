using CSV
using DataFrames
using PyPlot

include("SIRcasedeath.jl")
include("plotPout.jl")

# csvfile = CSV.File("covid_timeseries_UnitedKingdom.csv")
# dfuk = DataFrame(csvfile)
# dfuk[:,:active] = dfuk[:,:cum_cases] - dfuk[:,:cum_recover] - dfuk[:,:cum_deaths]
# @time out = mcmc(dfuk,[8,3,4],ones(6),6.67e7,10);
# @time out = mcmc(dfuk,[8,3,4],ones(6),6.67e7,100000);

# column 2, 3, 4: cum_cases, cum_deaths, cum_recover

# csvfile = CSV.File("covid_timeseries_Italy.csv")
# dfitaly = DataFrame(csvfile)
# dfitaly[:,:active] = dfitaly[:,:cum_cases] - dfitaly[:,:cum_recover] - dfitaly[:,:cum_deaths]

# @time out = mcmc(dfitaly,[5,6,7],[.2,.1,.1,.1,10.],6.05e7,100000);
# column 5, 6, 7: cum_cases, cum_deaths, cum_recover

# csvfile = CSV.File("covid_timeseries_Hubei.csv")
# dfhubei = DataFrame(csvfile)
# dfhubei[:,:active] = dfhubei[:,:cum_cases] - dfhubei[:,:cum_recover] - dfhubei[:,:cum_deaths]
# out = 0 # reset
# @time out = mcmc(dfhubei,[11,5,6],ones(6),81000,Int(10));
# @time out = mcmc(dfhubei,[11,5,6],ones(6),81000,Int(30e5));
# # @time out = mcmc(dfhubei,[11,5,6],ones(6),1.0e5,Int(30e5));

csvfile = CSV.File("covid_timeseries_Korea, South.csv")
dfkorea = DataFrame(csvfile)
out = 0 # reset
@time out = mcmc(dfkorea,[5,3,4],ones(6),5e7,Int(10));
@time out = mcmc(dfkorea,[5,3,4],ones(6),5e7,Int(30e5));

# csvfile = CSV.File("covid_timeseries_New York.csv")
# dfny = DataFrame(csvfile)

# pout = out[2];
# plotPout(pout)