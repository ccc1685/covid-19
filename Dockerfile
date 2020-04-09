FROM jupyter/datascience-notebook
RUN julia -e 'Pkg.init()' && \
    julia -e 'Pkg.update()' && \
    julia -e 'Pkg.add("Arpack", "CSV", "DataFrames", "DelimitedFiles", "DifferentialEquations", "Distributed", "Distributions", "NLsolve", "PyPlot", "StochasticDiffEq")' && \
    # Precompile Julia packages \
    julia -e 'Pkg.build("Arpack", "CSV", "DataFrames", "DelimitedFiles", "DifferentialEquations", "Distributed", "Distributions", "NLsolve", "PyPlot", "StochasticDiffEq")'
RUN git clone https://github.com/ccc1685/covid-19
WORKDIR covid-19
RUN julia -p 8 SICRPoissonscriptBW.jl