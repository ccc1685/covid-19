# ODE SIR/SICR models
# variables are concentrations, X/N
# Z = 1-S


# SICRLQM ODE
# Cases are infective, mitigation applied
# 7+1
function sicrqm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q,mbase,mlocation = p
    beta *= mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

# SICR ODE
# Cases are noninfective
# 4+1 parameters
function sicr!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I)*(1-Z) - sigmac*I - sigma*I
	du[5] = dZ = beta*(I)*(1-Z)
end

# SICRq ODE
# Cases are quarantined with effectiveness q
# 5+1 parameters
function sicrq!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

# SICRf ODE
# Cases are quarantined with effectiveness q
# 5+1 parameters
function sicrf!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,f = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I)*(1-Z) - sigmac*I - f*sigma*I
	du[5] = dZ = beta*(I)*(1-Z)
end

# SICRqf ODE
# Cases are quarantined with factor q
# Cases recover or die with factor f
# 6+1 parameters
function sicrqf!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q,f = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

# Mitigation applied
#6+1
function sicrm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I)*(1-Z) - sigmac*I - sigma*I
	du[5] = dZ = beta*(I)*(1-Z)
end

# Mitigation applied
# 7+1
function sicrfm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,f,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I)*(1-Z) - sigmac*I - f*sigma*I
	du[5] = dZ = beta*(I)*(1-Z)

end

# 7+1
function sicrqm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

# 8+1
function sicrqfm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q,f,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

function sicrlqfm!(du,u,p,t)
	I,C = u
	beta,sigmac,sigma,q,f,mbase,mlocation,mrate,cmax,c50 = p
    beta *= mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)))
    sigmac *= 1 + cmax*t/(c50+t)
    du[1] = dI = beta*(I+q*C) - sigmac*I - f*sigma*I
	du[2] = dC = sigmac*I - sigma*C
end


function linearrd!(du,u,p,t)
	I,C = u
	beta,sigmac,sigmar,sigmad,q,f,mbase,mlocation,mrate,cmax,c50 = p
    beta *= mbase + (1-mbase)/(1 + exp(mrate*(t - mlocation)))
    sigmac *= 1 + cmax*t/(c50+t)
    du[1] = dI = beta*(I+q*C) - sigmac*I - f*(sigmar+sigmad)*I
	du[2] = dC = sigmac*I - (sigmar+sigmad)*C
end
