using Distributions

"""
	function mcmc(data,r,total::Int)

Metropolis-Hastings MCMC of the SIR ODE model assuming a Poisson likelihood
for the cumulative death rate and a noninformative prior
data = cumulative deaths starting from the first death
r = array of 4 parameters used in the model
total = number of MCMC steps
output = (time series of loglikelihoods, time series of parameters, maximum loglikelihood, max loglikelihood parameters)

"""

function mcmc(data,r,total::Int)

	chi = loss(data,r)
	accept = 0
	rt = copy(r)
	chiml = chi
	rml = copy(r)

	chiout = Array{Float64,1}(undef,total)
	rout = Array{Float64,2}(undef,total,4)
	for step in 1:total

		sampleparams!(rt,r)
		chit = loss(data,rt)

		# mesa prior for psi: P(psi) = 1/50, psi < 50, 1/psi, psi >= 50
		if rand() < exp(chi - chit + log(max(r[4],50)) - log(max(rt[4],50)))
			chi = chit
			r = copy(rt)
			accept += 1
			if chi < chiml
				chiml = chi
				rml = copy(r)
			end
		end
		chiout[step] = chi
		rout[step,:] = r
	end

	return chiout, rout, accept/total,chiml,rml

end


# generate MCMC step guess from current parameter value r using logNormal distirbution
# with scale parameter manually adjusted so approximately a third of guesses are accepted
function sampleparams!(rt,r)
	for i in eachindex(r)
		d = Distributions.LogNormal(log(r[i]),.02)
		rt[i] = rand(d)
	end
end

# loglikelihood of data given parameters at r using a Poisson likelihood function
function loss(data,r)
	prediction = model(r)
	start = max(round(Int,r[4]),0) + 1
	chi = 0
	for i in eachindex(data)
		# d = Distributions.Normal(prediction[start+i-1],sqrt(prediction[start+i-1]))
		# println(max(prediction[start+i-1],eps(Float64)))
		d = Distributions.Poisson(max(prediction[start+i-1],eps(Float64)))
		chi -= loglikelihood(d,[data[i]])
	end
	return chi
end

# model prediction of model for cumulative deaths
function model(r)
	out = oxford(0.,1/6.67e7,round(Int,r[4])+17,.01,r[1],r[2],r[3]*6.67e7,round(Int,r[4]))
	return out[1]
end

function model2(r)
	out = oxfordrk2(0.,1/6.67e7,round(Int,r[4])+17,.01,r[1],r[2],r[3]*6.67e7,round(Int,r[4]))
	return out[1]
end

function modelconstrained(r)
	out = oxfordrk2(0.,1/6.67e7,round(Int,r[4])+17,.01,2.25*r[2],r[2],r[3]*6.67e7,round(Int,r[4]))
	return out[1]
end


# Lourenco  model, Runge-Kutta 2 stepper
function oxfordrk2(zinit,yinit,total,dt,beta,sigma,rtheta,psi)

	z = zinit
	y = yinit

	totalN = Int(total/dt)

	zout = Array{Float64,1}(undef,Int(total+1))
	yout = similar(zout)
	dout = similar(zout)
	tout = similar(zout)

	zout[1] = zinit
	yout[1] = yinit
	tout[1] = 0.
	interval = 1

	t = 1
	# Runge Kutta 2 method
	for n = 1:totalN
		y1 = dt*(beta*y*(1-z)-sigma*y)/2
		z1 = dt*beta*y*(1-z)/2
		y2 = dt*(beta*(y+y1)*(1-z-z1)-sigma*y)
		z2 = dt*beta*(y+y1)*(1-z-z1)
		y += y2
		z += z2
		if n*dt%interval == 0
			t += 1
			zout[t] = z
			yout[t] = y
			dout[t] = rtheta*zout[max(t-psi,0)+1]
			tout[t] = (n-1)*dt
		end
	end
	return dout,zout,yout,tout
end

# Lourenco  model, Euler stepper
function oxford(zinit,yinit,total,dt,beta,sigma,rtheta,psi)

	z = zinit
	y = yinit

	totalN = Int(total/dt)

	zout = Array{Float64,1}(undef,Int(total+1))
	yout = similar(zout)
	tout = similar(zout)
	dout = zeros(total+1)

	zout[1] = zinit
	yout[1] = yinit
	tout[1] = 0.
	interval = 1

	t = 1
	# Euler method
	for n = 1:totalN
		y += dt*(beta*y*(1-z)-sigma*y)
		z += dt*beta*y*(1-z)
		if n*dt%interval == 0
			t += 1
			zout[t] = z
			yout[t] = y
			dout[t] = rtheta*zout[max(t-psi,0)+1]
			tout[t] = (n-1)*dt
		end
	end
	return dout,zout,yout,tout
end
