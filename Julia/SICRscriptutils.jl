# Utility functions for SICR scripts to run set of models in tuple modeltypes on region

using Distributed
using DelimitedFiles
using Dates
using LinearAlgebra
using StatsBase
using Statistics
using Distributions
using DifferentialEquations
using DataFrames
using CSV

const txtstr = ".txt"

include("SICRPoisson.jl")

function runmcmc(total,datestrin,datestrout,temp=1.,N=1e7,pinstr= "pl",modeltype= ["rqfmt"])

    println(total,' ',temp)
    println(pinstr)

    pathin, pathout = makepaths(datestrin,datestrout)
    datafiles,regions = getdatafiles("data")
    println(datestrin,"->",datestrout)

    pstatstr = pathout * "pstats" * txtstr
    fpstat = open(pstatstr,"w")
    for j in eachindex(datafiles)
        datafile = datafiles[j]
        df = DataFrame(CSV.File("data/" * datafile))
        println(regions[j])
        for i in eachindex(modeltype)
            p,pfit,cols,data = mcmcargs(modeltype[i],regions[j],df,pathin,pinstr)
            # println(modeltype[i])
            outtuple = @spawn mcmc(data,cols,p,pfit,N,total,linearrd!,temp)

            model = modeltype[i]
            region = regions[j]
            writemeasures(outtuple,model,region,pstatstr)
            writecovariances(outtuple,model,region,pathout)
        end
    end
end


function runmcmcMP(total,datestrin,datestrout,temp=1.,N=1e7,pinstr= "pl",modeltype= ["rqfmt"])

    println(total,' ',temp)
    println(pinstr)

    pathin, pathout = makepaths(datestrin,datestrout)
    datafiles,regions = getdatafiles("data")
    println(datestrin,"->",datestrout)

    out = Array{Future,2}(undef,length(modeltype),length(datafiles))

    for j in eachindex(datafiles)
        datafile = datafiles[j]
        df = DataFrame(CSV.File("data/" * datafile))
        println(regions[j])
        for i in eachindex(modeltype)
            p,pfit,cols,data = mcmcargs(modeltype[i],regions[j],df,pathin,pinstr)
            # println(modeltype[i])
            out[i,j] = @spawn mcmc(data,cols,p,pfit,N,total,linearrd!,temp)
        end
    end

    pstatstr = pathout * "pstats" * txtstr
    fpstat = open(pstatstr,"w")
    for j in eachindex(datafiles), i in eachindex(modeltype)
        outtuple = fetch(out[i,j])
        model = modeltype[i]
        region = regions[j]
        writemeasures(outtuple,model,region,pstatstr)
        writecovariances(outtuple,model,region,pathout)
    end
end


# retries list of data files in folder datafolder
function getdatafiles(datafolder)
    datafiles = readdir("data")
    datafiles = datafiles[occursin.("covid_timeseries",datafiles)]
    regions = Array{String,1}(undef,0)
    for datafile in datafiles
        push!(regions,replace(replace(datafile,".csv"=>""),"covid_timeseries_"=>""))
    end
    return datafiles,regions
end


# returns prior for parameters for original r and u models
# param order: p =  beta,sigmac,sigmar,sigmad,q,f,mbase,mlocation,mrate,cmax,c50
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

function makepaths(datestrin,datestrout)
    if ~isdir("Results")
        mkdir("Results")
    end
    pathin = "Results/$datestrin/"
    pathout = "Results/$datestrout/"
    if ~isdir(pathout)
        mkdir(pathout)
    end
    return pathin, pathout
end

# write out parameter and R0 statistics
function writemeasures(outtuple,name,region,pstatstr)
    pstats,R0 = measures(outtuple[2],outtuple[5])
    f = open(pstatstr,"a")
    writedlm(f,[region * "_" * name])
    writedlm(f,[outtuple[4] R0[1] R0[2] R0[3] R0[4]'])
    for ptup in pstats
        writedlm(f, [ptup[1] ptup[2] ptup[3] ptup[4]'])
    end
    close(f)
end

# write out max likelihood params
function writemaxll(outtuple,name,region,pathout)
    mlstr = pathout * "maxll" * "_" * region * "_" * name * txtstr
    f=open(mlstr,"w")
    f=open(mlstr,"a")
    writedlm(f,[outtuple[4];outtuple[5]])
    close(f)
end

# write out all loglikelihood mcmc samples
function writellsamples(outtuple,name,region,pathout)
    llstr = pathout * "llsamples" * "_" * region * "_" * name * txtstr
    f=open(llstr,"w")
    f=open(llstr,"a")
    writedlm(f,outtuple[1])
    close(f)
end

# write out last parameter in run
function writelastparam(outtuple,name,region,pathout)
    pstr = pathout * "pl" * "_" * region * "_" * name * txtstr
    f=open(pstr,"w")
    f=open(pstr,"a")
    writedlm(f,[outtuple[4];outtuple[2][end,:]])
    close(f)
end

# write out all parameters parameter in run
function writeparamsamples(outtuple,name,region,pathout)
    pstr = pathout * "paramsamples" * "_" * region * "_" * name * txtstr
    f=open(pstr,"w")
    f=open(pstr,"a")
    for i in 1:size(outtuple[2],1)
        writedlm(f,[outtuple[2][i,:]])
    end
    close(f)
end

function writecovariances(outtuple,name,region,pathout)
    pstr = pathout * "covcor" * "_" * region * "_" * name * ".csv"
    f=open(pstr,"w")
    f=open(pstr,"a")
    writedlm(f,cov(outtuple[2]),',')
    writedlm(f,cor(outtuple[2]),',')
    close(f)
end

# output arguments used in mcmc function
function mcmcargs(name,region,df,pathin,pinstr)
    colsr = [5,6,7]
    colsu = [5,6]
    if occursin("u",name)
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
    return p,pfit,cols,data
end
