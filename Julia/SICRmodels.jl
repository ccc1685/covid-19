# ODE SIR/SICR models
# variables are concentrations, X/N
# Z = 1-S


# Classic SIR model, all infected our cases
# 4 parameters
function sir!(du,u,p,t)
C,D,R,Z = u
beta,sigmad,sigmar = p
du[1] = dC = beta*(1-Z)*C - (sigmad + sigmar)*C
du[2] = dD = sigmad*C
du[3] = dR = sigmar*C
du[4] = dZ = beta*(1-Z)*C
end


# SICR ODE model using DifferentialEquations.jl

# Cases are not quarantined
# 5 parameters
function sicr!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigmad,sigmar = p
	du[1] = dC = sigmac*I - (sigmar + sigmad)*C
	du[2] = dD = sigmad*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+C)*(1-Z) - (sigmac + sigmar + sigmad)*I
	du[5] = dZ = beta*(I+C)*(1-Z)
end

# SICRq ODE
# Cases are quarantined with effectiveness q
# 6 parameters
function sicrq!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigmad,sigmar,q = p
	du[1] = dC = sigmac*I - (sigmar + sigmad)*C
	du[2] = dD = sigmad*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - (sigmac + sigmar + sigmad)*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end

# SICRq ODE
# Cases are quarantined with factor q
# Cases recover or die with factor f
# 6 parameters
function sicrqf!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigmad,sigmar,q,f = p
	du[1] = dC = sigmac*I - f*(sigmar + sigmad)*C
	du[2] = dD = sigmad*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - (sigmac + sigmar + sigmad)*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)
end


function sicrqmitigate!(du,u,p,t)
	C,D,R,I,Z = u
	beta,sigmac,sigmad,sigmar,q,f = p
    if C > 500
        beta *= f
    end
	du[1] = dC = sigmac*I - (sigmar + sigmad)*C
	du[2] = dD = sigmad*C
	du[3] = dR = sigmar*C
	du[4] = dI = beta*(I+q*C)*(1-Z) - (sigmac + sigmar + sigmad)*I
	du[5] = dZ = beta*(I+q*C)*(1-Z)

end
