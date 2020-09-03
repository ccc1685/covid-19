using DelimitedFiles

datafiles = readdir("/Users/carsonc/github/covid-sicr/data")
datafiles = readdir("/Users/carsonc/github/VaxBRAModelEPI/USStates/")

datafiles = datafiles[occursin.("covidtimeseries_US_",datafiles)]

regions = Array{String,1}(undef,0)
for datafile in datafiles
    push!(regions,replace(replace(datafile,".csv"=>""),"covidtimeseries_"=>""))
end

# models = ["fulllinearmodel","reducedlinearmodelq0","reducedlinearmodelq0ctime","reducedlinearmodelNegBinom","nonlinearmodelq0ctime","nonlinearmodel"]
# models = ["nonlinearmodelq0ctime"]
# models = ["nonlinearmodel"]
# models = ["linearm2R","linearmqc2R"]
models = ["SICRM","SICRLM","SICRMQC","SICRLMQC","SICRM2R","SICRMQC2R","SICRLM2R","SICRLMQC2R"]
models = ["SICRMQ","SICRLMQ","SICRMQ2R","SICRLMQ2R"]
models = ["SICRMQC2R2DX2"]
models = ["SICR2HV"]

for modelname in models

    iter = "-it=10000"
    warm = "-wm=6000"
    format = "-f=1"
    fitpath = "-fp='/data/carsonc/covid-19/fitUS06-27-20V/'"
    modelpath = "-mp='/home/carsonc/covid-sicr/models/'"
    # datapath = "-dp='/home/carsonc/covid-sicr/data/'"
    datapath = "-dp='/home/carsonc/VaxBRAModelEPI/USStates/'"
    ad = "-ad=.85"
    ft = "-ft=1"
    ld = "-ld='06/27/20'"
    runfile = "&& python /home/carsonc/covid-sicr/scripts/runV.py"


    swarmname = modelname * ".swarm"

    f = open(swarmname,"w")

    for region in regions
        # init = "-i='/data/carsonc/covid-19/fit10/" * modelname * "_" * region * ".pkl'"
        init = ""
        regionarg = "-r='" * region * "'"
        writedlm(f,["source /data/carsonc/conda/etc/profile.d/conda.sh \\" ])
        writedlm(f,[ "&& conda activate covid-19 \\" ])
        writedlm(f,[ runfile modelname regionarg modelpath datapath fitpath iter warm format ad ft ld])
        # writedlm(f,[ "/data/carsonc/conda/envs/covid-19/bin/python run.py" modelname region fitpath iter warm format ])
    end

    close(f)
end
