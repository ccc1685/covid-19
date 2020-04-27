using CSV
using DataFrames

function createUSDF()

    csvfile = CSV.File("states_daily_4pm_et.csv")
    df = DataFrame(csvfile)
    idx = typeof.(df[:,:recovered]) .== Int
    dfrecovered = df[idx,:]
    states = unique(dfrecovered,:state)[:,:state]
    
    for i = 1:length(states)
        dfstatetmp = df[df[:,:state] .== states[i],:]
        dfstate = select(dfstatetmp,[:date,:positive,:death,:recovered])
        for i = 1:size(dfstate,2)
            idx = typeof.(dfstate[:,i]) .== Int
            dfstate[.!idx,i] .= 0
        end
        rename!(dfstate,[Symbol("date"), Symbol("cum_case"), Symbol("cum_death"), Symbol("cum_recover")])
        dfstate[:,:active] = dfstate[:,:cum_case] - dfstate[:,:cum_death] - dfstate[:,:cum_recover]

        sort!(dfstate,:cum_case)

        CSV.write("covid_timeseries_US_" * states[i] * ".csv",dfstate)
    end

end