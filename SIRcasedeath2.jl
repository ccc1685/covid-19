using Distributions
using DifferentialEquations
using DataFrames
using CSV
using PyPlot

"""
	function mcmc(data,cols,p,N,total::Int)

Metropolis-Hastings MCMC of the SIR ODE model assuming a Poisson likelihood
for the cumulative number of cases and deaths, and a noninformative prior
data = DataFrame of
cols = [active cases, deaths, recovered cases]
p[1:end-1] = array of parameters used in the model
p[end] = number infected on day 0 of data (Jan 22, 2020)
N = population of data region
total = number of MCMC steps
output = (loglikelihoods, parameters, maximum loglikelihood, max loglikelihood parameters)

"""
function mcmc(data::DataFrame,cols::Array,p::Array,N::Float64,total::Int,sicrfn! = sicr!)

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
	for i in eachindex(p)
		d = Distributions.LogNormal(log(p[i]),.002)
		pt[i] = rand(d)
	end
end

# loglikelihood of data given parameters at r using a Poisson likelihood function
function ll(data,cols,p,N,sicrfn!)
	firstday = findfirst(data[:,cols[1]] .>0)
	lastday = length(data[:,cols[1]])
	tspan = (firstday,lastday)
	uin = [data[firstday,cols[1]],data[firstday,cols[2]],data[firstday,cols[3]]]/N
	prediction,days = model(p,float.(uin),float.(tspan),N,sicrfn!)
	chi = 0
	for set in 1:3
		for i in eachindex(days)
			d = Distributions.Poisson(max(prediction[set,i],2.))
			chi -= loglikelihood(d,[data[days[i],cols[set]]])
		end
	end
	return chi
end

# produce arrays of cases and deaths
function model(pin,uin,tspan,N,sicrfn!)
	p = pin[1:end-1]
	u0 = vcat(uin,[(p[2]+p[3])/p[1]*log(1-pin[end]/N)+pin[end]/N + 1/N,pin[end]/N,0.])
	# u0 = [uin,p[end],p[end]*exp(p[1]/(p[1]-p[2])),0.]
	prob = ODEProblem(sicrfn!,u0,tspan,p)
	sol = solve(prob,saveat=1.)
	return sol[1:3,:]*N, Int.(sol.t)
end

# SICR ODE model using Differential Equations
function sicr!(du,u,p,t)
	# variables are concentrations X/N
	C,D,R,I,Z,U = u
	beta,sigmac,sigmau,gammad,gammar = p
	du[1] = dC = sigmac*I - gammad*C - gammar*C
	du[2] = dD = gammad*C
	du[3] = dR = gammar*C
	du[4] = dI = beta*I*(1-Z) - sigmac*I - sigmau*I
	du[5] = dZ = beta*I*(1-Z)
	du[6] = dU = sigmau*I
end

# SICR ODE model using Differential Equations
function sicrc!(du,u,p,t)
	# variables are concentrations X/N
	C,D,R,I,Z,U = u
	beta,sigmac,sigmau,gammad,gammar = p
	du[1] = dC = sigmac*I - gammar*C - gammad*C
	du[2] = dD = gammad*C
	du[3] = dR = gammar*C
	du[4] = dI = beta*(I+C)*(1-Z) - sigmac*I - sigmau*I
	du[5] = dZ = beta*(I+C)*(1-Z)
	du[6] = dU = sigmau*I
end


function makehistograms(out)
	for i in 1:size(out)[2]
		subplot(2,3,i)
		hist(out[:,i],bins=100,density=true)
		title("param$(i)",fontsize=15)
	end
	tight_layout()
end

function modelsol(data,cols,p,N,sicrfn!)
	firstday = findfirst(data[:,cols[1]] .>0)
	lastday = length(data[:,cols[1]])
	tspan = (firstday,lastday)
	uin = [data[firstday,cols[1]],data[firstday,cols[2]],data[firstday,cols[3]]]/N
	prediction,days = model(p,float.(uin),float.(tspan),N,sicrfn!)
	return prediction', data[days,cols]

end



# loglikelihood of data given parameters at r using a Poisson likelihood function
function llnew(data,cols,p,N)
	total = length(data[!,cols[1]])-1
	prediction = modelnew(p,total,N)
	chi = 0
	for set in 1:3
		for i in 1:total
			d = Distributions.Poisson(max(prediction[i,set],2.))
			chi -= loglikelihood(d,[data[i,cols[set]]])
		end
	end
	return chi
end
