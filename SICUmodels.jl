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
# 6 parameters
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
# 6 parameters
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
# 6 parameters
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
# 6 parameters
function sicuqf!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,q,f = p
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end

# Mitigation applied
function sicum!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,f,ml,mr = p
    beta *= 1/(1 + exp(mr*(t- 35 - ml)))
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I)*(1-Z) - sigmac*I + f*sigma*I
	du[4] = dZ = beta*(I)*(1-Z)
end

# Mitigation applied after C > 500
function sicufm!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,f,m = p
    if C > 500
        beta *= m
    end
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I)*(1-Z) - sigmac*I + f*sigma*I
	du[4] = dZ = beta*(I)*(1-Z)
end

# Mitigation applied after C > 500
function sicuqfm!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigma,q,f,m = p
    if C > 500
        beta *= m
    end
	du[1] = dC = sigmac*I - sigma*C
	du[2] = dU = sigma*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I - f*sigma*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end
