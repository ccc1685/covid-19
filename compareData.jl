using CSV
using DataFrames
using PyPlot


# csvfile = CSV.File("covid_timeseries_UnitedKingdom.csv")
# csvfile = CSV.File("covid_timeseries_Italypre.csv")
csvfile = CSV.File("covid_timeseries_Hubei.csv")
dfuk1 = DataFrame(csvfile)
dfuk1[:,:active] = dfuk1[:,:cum_cases] - dfuk1[:,:cum_recover] - dfuk1[:,:cum_deaths]

# csvfile = CSV.File("covid_timeseries_United Kingdom.csv")
# csvfile = CSV.File("covid_timeseries_Italy.csv")
csvfile = CSV.File("covid_timeseries_China.csv")
dfuk2 = DataFrame(csvfile)


figure()
subplot(3,1,1)
plot(dfuk1[:,6],color="k",linewidth=5)
plot(dfuk2[:,3],color="r",linewidth=1.5)
title("cum death",fontsize=15)

subplot(3,1,2)
plot(dfuk1[:,7],color="k",linewidth=5)
plot(dfuk2[:,4],color="r",linewidth=1.5)
title("cum recover",fontsize=15)

subplot(3,1,3)
plot(dfuk1[:,11],color="k",linewidth=5)
plot(dfuk2[:,5],color="r",linewidth=1.5)
title("active cases",fontsize=15)
tight_layout()

