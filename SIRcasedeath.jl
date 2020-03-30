using Distributions
using DifferentialEquations
using DataFrames
using CSV
using PyPlot

"""
	function mcmc(data,cols,p,Population,total::Int)

Metropolis-Hastings MCMC of the SIR ODE model assuming a Poisson likelihood
for the cumulative number of cases and deaths, and a noninformative prior
data = DataFrame of
cols = [active cases, deaths, recovered cases]
p[1:end-1] = array of parameters used in the model
p[end] = day of first infection relative to day 0 of data (Jan 22, 2020)
Population = population of region in data
total = number of MCMC steps
output = (loglikelihoods, parameters, maximum loglikelihood, max loglikelihood parameters)

"""
function mcmc(data::DataFrame,cols::Array,p::Array,Population::Float64,total::Int)

	chi = ll(data,cols,p,Population)
	accept = 0
	pt = copy(p)
	chiml = chi
	pml = copy(p)

	chiout = Array{Float64,1}(undef,total)
	pout = Array{Float64,2}(undef,total,length(p))

	for step in 1:total
		sampleparams!(pt,p)
		chit = ll(data,cols,pt,Population)
		if rand() < exp(chi - chit)
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
	for i in 1:length(p) - 1
		d = Distributions.LogNormal(log(p[i]),.002)
		pt[i] = rand(d)
	end
	pt[end] = p[end] + .05*randn()
end

# loglikelihood of data given parameters at r using a Poisson likelihood function
function ll(data,cols,p,Population)
	total = length(data[!,cols[1]])-1
	prediction = model(p,total,Population)
	chi = 0
	for set in 1:3
		for i in 1:total
			d = Distributions.Poisson(max(prediction[i,set],1.))
			chi -= loglikelihood(d,[data[i,cols[set]]])
		end
	end
	return chi
end

# produce arrays of cases and deaths
function model(pin,total,Population)
	u0 = [0.,0.,0.,1/Population,0.,0.]
	p = pin[1:end-1]
	if pin[end] < total - 1
		start = ceil(Int,pin[end])
		tspan = (pin[end],total-1)
		prob = ODEProblem(sirc!,u0,tspan,p)
		sol = solve(prob,saveat=collect(start:total-1))
		# align solutions to time 0 of data and reorder rows and cols
		prediction = zeros(total,3)
		for set in 1:3
			prediction[max(0,start)+1:end,set] .= sol[set,max(0,-start)+1:end]
		end
		return prediction*Population
	else
		return zeros(total,3)
	end
end

# SIRC ODE model using Differential Equations
function sirc!(du,u,p,t)
	C,D,R,Y,Z,U = u
	beta,sigmac,sigmar,gammad,gammar = p
	du[1] = dC = sigmac*Y - gammar*C - gammad*C
	du[2] = dD = gammad*C
	du[3] = dR = gammar*C
	du[4] = dY = beta*Y*(1-Z) - sigmac*Y - sigmar*Y
	du[5] = dZ = beta*Y*(1-Z)
	du[6] = dU = sigmar*Y
end
