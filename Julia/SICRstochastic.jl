# ODE SIR/SICR models
# variables are concentrations, X/N
# Z = 1-S


function sir(T,N,beta,sigmaR,sigmaD,a,b)

	dI = zeros(Int,T)
	dR = zeros(Int,T)
	dD = zeros(Int,T)
	I = zeros(Int,T)

	S = N
	I[1] = 10
	for i = 2:T
		d = Binomial(S,beta*S*I[i-1]/N^2)
		dI[i] = rand(d)
		d = Binomial(I[i-1],sigmaR*I[i-1]/N)
		dR[i] = rand(d)
		d=Gamma(a,b)
		ind = round(Int,rand(d))
		ind = max(i-1-ind,1)
		d = Binomial(I[i-1],sigmaD*I[ind]/N)
		dD[i] = rand(d)
		I[i] = I[i-1] + dI[i] - dR[i] - dD[i]
		S -=  dI[i]
	end

	return dI, dR, dD

end

function sicr(T,N,beta,sigmaI,sigmaC,sigmaR,sigmaD,a,b)

	dI = zeros(Int,T)
	dC = zeros(Int,T)
	dR = zeros(Int,T)
	dD = zeros(Int,T)
	dC = zeros(Int,T)
	C = zeros(Int,T)
	S = N
	I = 10
	for i = 2:T
		d = Binomial(S,beta*S*I/N^2)
		dIp = rand(d)
		d = Binomial(I,sigmaI*I/N)
		dIm = rand(d)
		d = Binomial(I,sigmaC*I/N)
		dC[i] = rand(d)
		d = Binomial(C[i-1],sigmaR*C[i-1]/N)
		dR[i] = rand(d)
		d=Gamma(a,b)
		ind = round(Int,rand(d))
		ind = max(i-1-ind,1)
		d = Binomial(C[i-1],sigmaD*C[ind]/N)
		dD[i] = rand(d)
		C[i] = C[i-1] + dC[i] - dR[i] - dD[i]
		I += dIp - dIm - dC[i]
		S -=  dIp
	end

	return dC, dR, dD

end

function sicr(T,N,beta,sigmaI,sigmaC,sigmaR,sigmaD)

	dI = zeros(Int,T)
	dC = zeros(Int,T)
	dR = zeros(Int,T)
	dD = zeros(Int,T)
	dC = zeros(Int,T)
	C =1
	R = 0
	S = N
	I = 10
	for i = 2:T
		d = Binomial(S,beta*S*I/N^2)
		dIp = rand(d)
		d = Binomial(I,sigmaI*I/N)
		dIm = rand(d)
		d = Binomial(I,sigmaC*I/N)
		dC[i] = rand(d)
		if C > 0
			d = Binomial(C,sigmaR*C/N)
			dR[i] = rand(d)
		else
			dC[i] = 0
		end
		if R > 0
			d = Binomial(R,sigmaD*R/N)
			dD[i] = rand(d)
		else
			dD[i] =0
		end
		C +=  dC[i] - dR[i] - dD[i]
		R += dR[i] - dD[i]
		I += dIp - dIm - dC[i]
		S -=  dIp
	end

	return dC, dR, dD

end
