using CSV
using DataFrames

function loadData()
    startDate = "1/22/20"
    endDate = "3/20/20"
    roi = [(0,"Italy"), (0,"Spain"), ("Hubei","China"), (0,"Germany"), (0,"France"), (0,"Iran"), (0,"United Kingdom"), (0,"Switzerland"), (0,"Turkey"), (0,"Belgium"), (0,"Netherlands"), (0,"Austria"), (0,"Korea, South")] #(Province/State, Country/Region); Province/State=0 if data empty

    for i = 1:length(roi)
        buildDataFrame(roi[i],startDate,endDate)
    end

end

function buildDataFrame(roi,startDate,endDate)
    csvfile_confirmed = CSV.File("time_series_covid19_confirmed_global.csv")
    csvfile_deaths = CSV.File("time_series_covid19_deaths_global.csv")
    csvfile_recovered = CSV.File("time_series_covid19_recovered_global.csv")
    dfconfirmed = DataFrame(csvfile_confirmed)
    dfdeaths = DataFrame(csvfile_deaths)
    dfrecovered = DataFrame(csvfile_recovered)
        
    # region of interest
    if roi[1] == 0
        rowind_confirmed = .!(typeof.(dfconfirmed[:,Symbol("Province/State")]) .== String) .& (dfconfirmed[:,Symbol("Country/Region")] .== roi[2])
        rowind_deaths = .!(typeof.(dfdeaths[:,Symbol("Province/State")]) .== String) .& (dfdeaths[:,Symbol("Country/Region")] .== roi[2])
        rowind_recovered = .!(typeof.(dfrecovered[:,Symbol("Province/State")]) .== String) .& (dfrecovered[:,Symbol("Country/Region")] .== roi[2])
    else
        rowind_confirmed = (dfconfirmed[:,Symbol("Province/State")] .== roi[1]) .& (dfconfirmed[:,Symbol("Country/Region")] .== roi[2])
        rowind_deaths = (dfdeaths[:,Symbol("Province/State")] .== roi[1]) .& (dfdeaths[:,Symbol("Country/Region")] .== roi[2])
        rowind_recovered = (dfrecovered[:,Symbol("Province/State")] .== roi[1]) .& (dfrecovered[:,Symbol("Country/Region")] .== roi[2])
    end
    dfconfirmed_roi = dfconfirmed[rowind_confirmed,:];
    dfdeaths_roi = dfdeaths[rowind_deaths,:];
    dfrecovered_roi = dfrecovered[rowind_recovered,:];

    # time period of interest
    colind = collect(1:size(dfconfirmed_roi,2))
    startInd = colind[names(dfconfirmed_roi) .== Symbol(startDate)][1]
    endInd = colind[names(dfconfirmed_roi) .== Symbol(endDate)][1]

    # tailored to roi and toi 
    nameInd = names(dfconfirmed_roi)[startInd:endInd]
    dfconfirmed_roi_toi = dfconfirmed_roi[startInd:endInd]
    dfdeaths_roi_toi = dfdeaths_roi[startInd:endInd]
    dfrecovered_roi_toi = dfrecovered_roi[startInd:endInd]

    # combine confirm, death, recover, active
    dflength = size(dfconfirmed_roi_toi,2)
    dfroi = DataFrame()
    dfroi.date = nameInd
    dfroi.cum_case = stack(dfconfirmed_roi_toi,1:dflength)[:,:value]
    dfroi.cum_death = stack(dfdeaths_roi_toi,1:dflength)[:,:value]
    dfroi.cum_recover = stack(dfrecovered_roi_toi,1:dflength)[:,:value]
    dfroi.active = dfroi.cum_case - dfroi.cum_death - dfroi.cum_recover

    CSV.write("covid_timeseries_" * roi[2] * ".csv",dfroi)

end