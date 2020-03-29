using Distributions
using DifferentialEquations
using DataFrames

"""
	function mcmc(data,cols,p,Population,total::Int)

Metropolis-Hastings MCMC of the SIR ODE model assuming a Poisson likelihood
for the cumulative number of cases and deaths, and a noninformative prior
data = DataFrame of
cols = [cumulative cases, deaths, recovered]
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
		d = Distributions.LogNormal(log(p[i]),.02)
		pt[i] = rand(d)
	end
	pt[end] = round(p[end] + randn())
end

# loglikelihood of data given parameters at r using a Poisson likelihood function
function ll(data,cols,p,Population)
	total = length(data[!,cols[1]])-1
	if p[end] > total - 2
		chi = 1e6
	else
		prediction = model(p,total,Population)
		chi = 0
		start = Int(abs(p[end]))
		if p[end] > 0.
			for set in 1:3
				if data[start+1, cols[set]] > 0
					chi = 1e6
				else
					for i in 2:length(prediction[set,:])
						d = Distributions.Poisson(max(Population*prediction[set, i],1e-100))
						chi -= loglikelihood(d,[data[start + i,cols[set]]])
					end
				end
			end
		else
			for set in 1:3
				for i in 2:length(data[!,cols[set]])-1
					d = Distributions.Poisson(max(Population*prediction[set, start + i],1e-100))
					chi -= loglikelihood(d,[data[i,cols[set]]])
				end
			end
		end
	end
	return chi
end

# produce arrays of cases and deaths
function model(pin,total,Population)
	u0 = [0.,0.,0.,1/Population,0.,0.]
	p = pin[1:end-1]
	tspan = (pin[end],total-1)
	prob = ODEProblem(sirc!,u0,tspan,p)
	sol = solve(prob,saveat=1)
	return sol
end

# SIR ODE model with new death model, Euler stepper
# function sir(zinit,yinit,r,P,total,dt=.01)
#
# 	z = zinit
# 	y = yinit
# 	d = 0.
#
# 	beta = r[1]
# 	sigmad = r[2]
# 	sigmar = r[3]
# 	f = r[4]
#
# 	totalN = Int(total/dt)
#
# 	zout = [zinit]
# 	yout = [yinit]
# 	tout = [0.]
# 	dout = [0.]
# 	cout = [0.]
# 	zout[1] = zinit
# 	yout[1] = yinit
# 	tout[1] = 0.
# 	interval = 1
#
# 	m = 0
# 	c = 0.
#
# 	# Euler method, run until at least one case appears
# 	while C*P < .5 && m < 1000
# 		step()
# 		m += 1
# 		Y += dt*(beta*y*(1-Z)-sigmad*Y - sigmar*Y)
# 		Z += dt*beta*Y*(1-Z)
# 		D += dt*sigmad*Y
# 		R += dt*sigmar*Y
# 		C += dt*f*Y
# 		CR += dt*f*R
# 		if m*dt%interval == 0
# 			push!(zout,1-Z)
# 			push!(yout,Y)
# 			push!(dout,P*D)
# 			push!(cout,P*C)
# 			push!(tout,m*dt)
# 		end
# 	end
# 	# Continue for total more days
# 	for n in m+1:m+totalN
# 		Y += dt*(beta*Y*(1-Z)-sigma*Y)
# 		Z += dt*beta*Y*(1-Z)
# 		D += dt*sigma*Y*psi/(1+psi)
# 		C += dt*f*Y
# 		CR += dt*f*R
# 		if n*dt%interval == 0
# 			push!(zout,Z)
# 			push!(yout,Y)
# 			push!(dout,P*D)
# 			push!(cout,P*D)
# 			push!(tout,n*dt)
# 		end
# 	end
# 	return cout,dout,zout,yout,tout
# end

function sirc!(du,u,p,t)
	C,D,CR,Y,Z,R = u
	beta,sigmad,sigmar,f = p
	du[1] = dC = f*Y
	du[2] = dD = sigmad*Y
	du[3] = dCR = f*R
	du[4] = dY = beta*Y*(1-Z) - sigmad*Y - sigmar*Y
	du[5] = dZ = beta*Y*(1-Z)
	du[6] = dR = sigmar*Y
end

function compare(p,total,Population)
	sol = model(p,total,Population)
	prediction = zeros(total,3)
	for set in 1:3
		prediction[max(0,Int(p[end]))+1:end,set] .= sol[set,max(0,-Int(p[end]))+1:end]
	end
	return prediction*Population
end
