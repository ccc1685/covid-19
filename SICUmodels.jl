# ODE SIU/SICU models
# variables are concentrations, X/N
# Z = 1-S
# U = are noninfectious include both recovered and dead

function sicuprep(data)
    df = data
    df[:,:uninfected] = df[:,:cum_recover] + df[:,:cum_death]
    return df
end

# Classic SIR model, all infected our cases
# 4 parameters
function scu!(du,u,p,t)
C,U,Z = u
beta,sigmau = p
du[1] = dC = beta*(1-Z)*C - sigmau*C
du[2] = dU = sigmau*C
du[3] = dZ = beta*(1-Z)*C
end

# SICR ODE model using DifferentialEquations.jl

# Cases are not quarantined
# 5 parameters
function sicuf!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigmau,f = p
	du[1] = dC = sigmac*I - sigmau*C
	du[2] = dU = sigmau*C
	du[3] = dI = beta*(I+C)*(1-Z) - sigmac*I + f*sigmau*I
	du[4] = dZ = beta*(I+C)*(1-Z)
end

# SICRq ODE
# Cases are quarantined with effectiveness q
# 6 parameters
function sicuq!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigmau,q = p
	du[1] = dC = sigmac*I - sigmau*C
	du[2] = dU = sigmau*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - (sigmac + sigmau)*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end

# SICRq ODE
# Cases are quarantined with factor q
# Cases recover or die with factor f
# 6 parameters
function sicuqf!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigmau,q,f = p
	du[1] = dC = sigmac*I - sigmau*C
	du[2] = dU = sigmau*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I + f*sigmau*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end

# Mitigation applied after C > 500
function sicuqfm!(du,u,p,t)
	C,U,I,Z = u
	beta,sigmac,sigmau,q,f,m = p
    if C > 500
        beta *= m
    end
	du[1] = dC = sigmac*I - sigmau*C
	du[2] = dU = sigmau*C
	du[3] = dI = beta*(I+q*C)*(1-Z) - sigmac*I + f*sigmau*I
	du[4] = dZ = beta*(I+q*C)*(1-Z)
end
