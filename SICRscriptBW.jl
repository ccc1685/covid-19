# Julia script to run set of models in tuple modeltypes on region
t0 = time()

@everywhere using DelimitedFiles
@everywhere using StatsBase
@everywhere using Dates
@everywhere using LinearAlgebra
@everywhere using StatsBase

const txtstr = ".txt"

@everywhere include("SICRnew.jl")

function getdatafiles(datafolder)
    datafiles = readdir("data")
    datafiles = datafiles[occursin.("covid_timeseries",datafiles)]
    regions = Array{String,1}(undef,0)
    for datafile in datafiles
        push!(regions,replace(replace(datafile,".csv"=>""),"covid_timeseries_"=>""))
    end
    return datafiles,regions
end

function getprior(modelname)
    if modelname == "r"
        return [.25,.1,.1,.1,.1]
    end
    if modelname == "rq" || modelname == "rf"
        return [.25,.1,.1,.1,.1,.1]
    end
    if modelname == "rm"
        return [.25,.1,.1,.1,.1,5.,.1]
    end
    if modelname == "rqm" || modelname == "rfm"
        return [.25,.1,.1,.1,.1,.1,5.,.1]
    end
    if modelname == "rqfm"
        return [.25,.1,.1,.1,.1,.1,.1,5.,.1]
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

name = Dict(sicr! => "r", sicrq! => "rq", sicrf! => "rf", sicrqf! => "rqf", sicrm! => "rm", sicrqm! => "rqm", sicrfm! => "rfm", sicrqfm! => "rqfm",sicu! => "u", sicuq! => "uq", sicuf! => "uf", sicuq! => "uq", sicuqf! => "uqf", sicum! => "um", sicuqm! => "uqm", sicufm! => "ufm", sicuqfm! => "uqfm")

datestrin = "2020-04-04"
datestrout = "2020-04-04"
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

colsr = [5,3,4]
colsu = [5,6]

N = 1e7
total = Int(1e7)
temp = 1.

modeltype = (sicr!,sicrq!,sicrf!,sicrm!,sicu!,sicuq!,sicuf!,sicum!)

out = Array{Future,2}(undef,length(modeltype),length(datafiles))

for j in eachindex(datafiles)
    datafile = datafiles[j]
    df = DataFrame(CSV.File("data/" * datafile))
    region = regions[j]
    println(region)

    for i in eachindex(modeltype)

        modrun = modeltype[i]

        if occursin("u",name[modeltype[i]])
            data = sicuprep(df)
            cols = colsu
        else
            data = df
            cols = colsr
        end

        infile = pathin * "maxll" * "_" * region * "_" * name[modrun] * txtstr

        if isfile(infile) && ~isempty(read(infile))
            p = readdlm(infile)[2:end]
        else
            p = getprior(name[modrun])
            println(name[modrun],": No prior")
        end

        out[i,j] = @time @spawn mcmc(data,cols,p,N,total,modrun,temp)

        println("maxll: ",out[i][4],", Accept: ",out[i][3])

    end
end


for j in eachindex(datafiles), i in eachindex(modeltype)

    outtuple = fetch(out[i,j])
    region = regions[j]
    modrun = modeltype[i]

    mlstr = pathout * "maxll" * "_" * region * "_" * name[modrun] * txtstr
    llstr = pathout * "llsamples" * "_" * region * "_" * name[modrun] * txtstr
    pstr = pathout * "pl" * "_" * region * "_" * name[modrun] * txtstr

    fml=open(mlstr,"w")
    fml=open(mlstr,"a")
    writedlm(fml,[outtuple[4];out[i][5]])
    close(fml)

    fchil=open(llstr,"w")
    fchil=open(llstr,"a")
    writedlm(fchil,outtuple[1])
    close(fchil)

    fpl=open(pstr,"w")
    fpll=open(pstr,"a")
    writedlm(fpl,outtuple[1])
    close(fpl)

end

println(" Time: ",time()-t0)
