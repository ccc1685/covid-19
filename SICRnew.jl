using Distributions
using DifferentialEquations
using DataFrames
using CSV
using PyPlot

include("SICRmodelsnew.jl")
include("SICUmodels.jl")

"""
	function mcmc(data,cols,p,N,total::Int,sicrfn!)

Metropolis-Hastings MCMC of the SIR ODE model assuming a Poisson likelihood
for the cumulative number of cases and deaths, and a noninformative prior
data = DataFrame of
cols = [active cases, deaths, recovered cases]
p[1:end-1] = array of parameters used in the model
p[end] = number infected on day 0 of data (Jan 22, 2020)
N = population of data region
total = number of MCMC steps
output = (loglikelihoods, parameters, maximum loglikelihood, max loglikelihood parameters)
sicrfn! = ODE file to use

"""
function mcmc(data::DataFrame,cols::Array,p::Array,N::Float64,total::Int,sicrfn! = sicr!,temp = 1.)

	chi = ll(data,cols,p,N,sicrfn!)
	accept = 0
	pt = copy(p)
	chiml = chi
	pml = copy(p)

	chiout = Array{Float64,1}(undef,total)
	pout = Array{Float64,2}(undef,total,length(p))

	for step in 1:total
		sampleparams!(pt,p)
		chit = ll(data,cols,pt,N,sicrfn!)
		if rand() < exp((chi - chit)/temp)
			chi = chit
			p = copy(pt)
			accept += 1
			if chi < chiml
				chiml = chi
				pml = copy(p)
			end
		end
		chiout[step] = chi
		pout[step,:] = p
	end
	return chiout, pout, accept/total, chiml, pml
end

# generate MCMC step guess from current parameter value r using logNormal distirbution for all
# parameters except start day with scale parameter manually adjusted so approximately a third of guesses are accepted
function sampleparams!(pt,p)
	for i in eachindex(p)
		d = Distributions.LogNormal(log(p[i]),.01)
		pt[i] = rand(d)
	end
end

# loglikelihood of data given parameters at r using a Poisson likelihood function
function ll(data,cols,p,N,sicrfn!)
	firstday = findfirst(data[:,cols[1]] .>0)
	lastday = length(data[:,cols[1]])
	tspan = (firstday,lastday)
	uin = uinitial(data,cols,firstday,N)
	prediction,days = model(p,uin,float.(tspan),N,sicrfn!)
	chi = 0
	for set in eachindex(cols)
		for i in eachindex(days)
			d = Distributions.Poisson(max(prediction[set,i],2.))
			chi -= loglikelihood(d,[data[days[i],cols[set]]])
		end
	end
	return chi
end

function uinitial(data,cols,firstday,N)
	uin = Array{Float64,1}(undef,length(cols))
	for i in eachindex(cols)
		uin[i] = data[firstday,cols[i]]/N
	end
	return uin
end

# produce arrays of cases and deaths
function model(pin,uin,tspan,N,sicrfn!)
	sol = modelsol(pin,uin,tspan,N,sicrfn!)
	return sol[1:3,:]*N, Int.(sol.t)
end

function modelsol(pin,uin,tspan,N,sicrfn!)
	p = pin[1:end-1]
	if sicrfn! == sir!
		u0 = vcat(uin,0.)
	else
		I0 = (p[1]- p[3])/p[2] * uin[1] + pin[end]/N
		Z0 = p[1]/p[2]*uin[1]
		u0 = vcat(uin,[I0,Z0])
	end
	# u0 = [uin,p[end],p[end]*exp(p[1]/(p[1]-p[2])),0.]
	prob = ODEProblem(sicrfn!,u0,tspan,p)
	solve(prob,saveat=1.)
end


function makehistograms(out)
	for i in 1:size(out)[2]
		subplot(2,3,i)
		hist(out[:,i],bins=100,density=true)
		title("param$(i)",fontsize=15)
	end
	tight_layout()
end

# returns predictions for C,R,D, matching data, and array of days
# solution array is in same row col order as data
# data columns have been ordered to match solutions.
function modelprediction(data,cols,p,N,sicrfn!)
	firstday = findfirst(data[:,cols[1]] .>0)
	lastday = length(data[:,cols[1]])
	sol = modelprediction(data,cols,p,N,sicrfn!,lastday)
	return sol[1:length(cols),:]'*N, data[firstday:end,cols], Int.(sol.t)
end

# returns solutions to model from day of first case to requested last day
function modelprediction(data,cols,p,N,sicrfn!,lastday)
	firstday = findfirst(data[:,cols[1]] .>0)
	tspan = (firstday,lastday)
	uin = uinitial(data,cols,firstday,N)
	modelsol(p,float.(uin),float.(tspan),N,sicrfn!)
end

# returns statistics for posteriors and other measures
function measures(psamples,pml)
	nparams = size(psamples,2)
	pout = Array{Tuple,1}(undef,nparams)
	for i in 1:nparams
		p = psamples[:,i]
		pout[i] = pml[i],mean(p),std(p),quantile(p,[.025,.5,.975])
	end
	R0 = pml[1]/(pml[2]+pml[3])
	return pout,R0
end
