# ODE SIU/SICU models
# variables are concentrations, X/N
# Z = 1-S
# U = are noninfectious include both recovered and dead

function sicuprep(data)
    df = data
    df[:,:uninfected] = df[:,:cum_recover] + df[:,:cum_death]
    return df
end

# SICUq ODE
# Cases are quarantined with effectiveness q
# 3+1 parameters
function sicu!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I)*(1-Z) - sigmac*I - sigma*I
	du[4] = dZ = beta*(I)*(1-Z)
end

# SICUq ODE
# Cases are quarantined with effectiveness q
# 4+1 parameters
function sicuq!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,q = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - sigma*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end

# SICUf ODE
# Cases are quarantined with factor q
# Cases recover or die with factor f
# 4+1 parameters
function sicuf!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,f = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I)*(1-Z) - sigmac*I - f*sigma*I
	du[4] = dZ = beta*(I)*(1-Z)
end

# SICUqf ODE
# Cases are quarantined with factor q
# Cases recover or die with factor f
# 5+1 parameters
function sicuqf!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,q,f = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end

# Mitigation applied
# 5 + 1 parameters
function sicum!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I)*(1-Z) - sigmac*I - sigma*I
	du[4] = dZ = beta*(I)*(1-Z)
end

# Mitigation applied after C > 500
# 6+1
function sicufm!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,f,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I)*(1-Z) - sigmac*I - f*sigma*I
	du[4] = dZ = beta*(I)*(1-Z)
end

# Mitigation applied after C > 500
# 6+1
function sicuqm!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,q,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - sigma*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end

# Mitigation applied after C > 500
# 7+1
function sicuqfm!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,q,f,mc,ml = p
    beta *= mc + (1-mc)/(1 + exp(t - 35. - ml))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end
