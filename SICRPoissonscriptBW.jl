# Julia script to run set of models in tuple modeltypes on region
# User sets total number of mcmc steps, folders for input and output and temperature
# run using:  julia -p #nprocs SICRPoissonscrptBW.jl

t0 = time()

using CSV
using DataFrames
using Dates
using DelimitedFiles
using DifferentialEquations
using Distributed
using Distributions
using LinearAlgebra
using StatsBase
using Statistics
using NLsolve
using StochasticDiffEq
@everywhere using CSV
@everywhere using DataFrames
@everywhere using Dates
@everywhere using DelimitedFiles
@everywhere using DifferentialEquations
@everywhere using Distributed
@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using StatsBase
@everywhere using Statistics
@everywhere using NLsolve
@everywhere using StochasticDiffEq


@everywhere include("SICRPoisson.jl")

const txtstr = ".txt"

function getdatafiles(datafolder)
    datafiles = readdir("data")
    datafiles = datafiles[occursin.("covid_timeseries",datafiles)]
    regions = Array{String,1}(undef,0)
    for datafile in datafiles
        push!(regions,replace(replace(datafile,".csv"=>""),"covid_timeseries_"=>""))
    end
    return datafiles,regions
end

# beta,sigmac,sigmar,sigmad,q,f,mbase,mlocation,mrate,cmax,c50 = p
function getpfit(modelname)
    if modelname == "r"
        return  [1;2;3;4;12]
    end
    if modelname == "rq"
        return [1;2;3;4;5;12]
    end
    if modelname == "rf"
        return [1;2;3;4;6;12]
    end
    if modelname == "rqf"
        return [1;2;3;4;5;6;12]
    end
    if modelname == "rm"
        return [1;2;3;4;7;8;9;12]
    end
    if modelname == "rqm"
        return [1;2;3;4;5;7;8;9;12]
    end
    if modelname == "rfm"
        return [1;2;3;4;6;7;8;9;12]
    end
    if modelname == "rqfm"
        return [1;2;3;4;5;6;7;8;9;12]
    end
    if modelname == "rmt"
        return [1;2;3;4;5;6;7;8;9;10;11;12]
    end
    if modelname == "rqmt" || modelname == "rfm"
        return [1;2;3;4;5;6;7;8;9;10;11;12]
    end
    if modelname == "rfmt"
        return [1;2;3;4;5;6;7;8;9;10;11;12]
    end
    if modelname == "rqfmt"
        return [1;2;3;4;5;6;7;8;9;10;11;12]
    end
    if modelname == "u"
        return [.25,.1,.1,.1]
    end
    if modelname == "uq" || modelname == "uf"
        return [.25,.1,.1,.1,.1]
    end
    if modelname == "um"
        return [.25,.1,.1,.1,5.,.1]
    end
    if modelname == "uqm" || modelname == "ufm"
        return [.25,.1,.1,.1,.1,5.,.1]
    end
    if modelname == "uqfm"
        return [.25,.1,.1,.1,.1,.1,5.,.1]
    end
end

# name = Dict(sicr! => "r", sicrq! => "rq", sicrf! => "rf", sicrqf! => "rqf", sicrm! => "rm", sicrqm! => "rqm", sicrfm! => "rfm", sicrqfm! => "rqfm",sicu! => "u", sicuq! => "uq", sicuf! => "uf", sicuq! => "uq", sicuqf! => "uqf", sicum! => "um", sicuqm! => "uqm", sicufm! => "ufm", sicuqfm! => "uqfm")

datestrin = "2020-04-08linear"
datestrout = "2020-04-08linear"
println(datestrin,"->",datestrout)
if ~isdir("Results")
    mkdir("Results")
end
pathin = "Results/$datestrin/"
pathout = "Results/$datestrout/"
if ~isdir(pathout)
    mkdir(pathout)
end

datafiles,regions = getdatafiles("data")

colsr = [5,6,7]
colsu = [5,6]

N = 1e7
total = Int(2.e3)
temp = 1.
const pinstr = "pl"

println(total,' ',temp)
println(pinstr)

# modeltype = (sicr!,sicrq!,sicrf!,sicrm!,sicu!,sicuq!,sicuf!,sicum!)
# modeltype = (sicr!,sicrq!,sicrf!,sicrfq!,sicrfqm!,sicurfqmt!)
modeltype = ["rqfmt"]

pstatstr = pathout * "pstats" * txtstr
fpstat = open(pstatstr,"w")

out = Array{Future,2}(undef,length(modeltype),length(datafiles))

for j in eachindex(datafiles)
    datafile = datafiles[j]
    df = DataFrame(CSV.File("data/" * datafile))
    region = regions[j]
    println(region)

    for i in eachindex(modeltype)

        name = modeltype[i]

        if occursin("u",modeltype[i])
            data = sicuprep(df)
            cols = colsu
        else
            data = df
            cols = colsr
        end

        infile = pathin * pinstr * "_" * region * "_" * name * txtstr

        if isfile(infile) && ~isempty(read(infile))
            p = readdlm(infile)[2:end]
            pfit = getpfit(name)
        else
            pfit = getpfit(name)
            p = [.25,.1,.01,.01,.01,1.,.1,1.,1.,.1,10.,1.]
            println(name,": No prior")
        end

        out[i,j] = @spawn mcmc(data,cols,p,pfit,N,total,linearrd!,temp)
        # println(name,": maxll: ",outtuple[4],", Accept: ",outtuple[3])
    end
end

for j in eachindex(datafiles), i in eachindex(modeltype)

    outtuple = fetch(out[i,j])

    name = modeltype[i]
    region = regions[j]

    mlstr = pathout * "maxll" * "_" * region * "_" * name * txtstr
    llstr = pathout * "llsamples" * "_" * region * "_" * name * txtstr
    pstr = pathout * "pl" * "_" * region * "_" * name * txtstr

    pstats,R0 = measures(outtuple[2],outtuple[5])
    fpstat = open(pstatstr,"a")
    writedlm(fpstat,[region * "_" * name])
    writedlm(fpstat,[outtuple[4] R0[1] R0[2] R0[3] R0[4]'])
    for ptup in pstats
        writedlm(fpstat, [ptup[1] ptup[2] ptup[3] ptup[4]'])
    end
    close(fpstat)

    fml=open(mlstr,"w")
    fml=open(mlstr,"a")
    writedlm(fml,[outtuple[4];outtuple[5]])
    close(fml)

    fchil=open(llstr,"w")
    fchil=open(llstr,"a")
    writedlm(fchil,outtuple[1])
    close(fchil)

    fpl=open(pstr,"w")
    fpll=open(pstr,"a")
    writedlm(fpl,[outtuple[4];outtuple[2][end,:]])
    close(fpl)

end

println(" Time: ",time()-t0)
