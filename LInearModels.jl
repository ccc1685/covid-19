# Mitigation applied after C > 500
# 7+1
function linearu!(du,u,p,t)
	I,C = u
	beta,sigmac,sigma,q,f,mbase,mlocation,mrate,fmax,f50 = p
    beta *= mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)))
    f *= 1 + fmax*t/(f50+t)
    du[1] = dI = beta*(I+q*C) - sigmac*I - f*sigma*I
	du[2] = dC = sigmac*I - sigma*C
end
function linearu!(du,u,p,t)
	I,C,S = u
	beta,sigmac,sigma,q,f,mbase,mlocation,mrate,fmax,f50 = p
    beta *= mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)))
    f *= 1 + fmax*t/(f50+t)
    du[1] = dI = beta*(I+q*C)*S - sigmac*I - f*sigma*I
	du[2] = dC = sigmac*I - sigma*C
    du[3] = dI = beta*(I+q*C)*S
end

function linearrd!(du,u,p,t)
	I,C = u
	beta,sigmac,sigmar,sigmad,q,f,mbase,mlocation,mrate,fmax,f50 = p
    beta *= mbase + mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)))
    f *= 1 + fmax*t/(f50+t)
    du[1] = dI = beta*(I+q*C) - sigmac*I - f*(sigmar+sigmad)*I
	du[2] = dC = sigmac*I - (sigmar+sigmad)*C
end

function nonlinearrd!(du,u,p,t)
	I,C,S = u
	beta,sigmac,sigmar,sigmad,q,f,mbase,mlocation,mrate,fmax,f50 = p
    beta *= mbase + mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)))
    f *= 1 + fmax*t/(f50+t)
    du[1] = dI = beta*(I+q*C)*S - sigmac*I - f*(sigmar+sigmad)*I
	du[2] = dC = sigmac*I - (sigmar+sigmad)*C
    du[3] = dS = -beta*(I+q*C)*S
end
