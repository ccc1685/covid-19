# ODE SIR/SICR models
# variables are concentrations, X/N
# Z = 1-S

# SICR ODE
# Cases are noninfective
# 6 parameters
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
# 6 parameters
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
# 6 parameters
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
# 6 parameters
function sicrqf!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q,f = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

function sicrm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35 - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I)*(1-Z) - sigmac*I - sigma*I
	du[5] = dZ = beta*(I)*(1-Z)
end

# Mitigation applied
function sicrfm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,f,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35 - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I)*(1-Z) - sigmac*I - f*sigma*I
	du[5] = dZ = beta*(I)*(1-Z)

end

function sicrqm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35 - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

function sicrqfm!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigma,sigmar,q,f,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35 - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dD = (sigma - sigmar)*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end
