for scenario in ["warmup", "baseline", "iterative", "partition", "filter"]
    push!(ARGS, realpath(joinpath(@__DIR__, "config", "config_$scenario.jl")))
    include(joinpath(@__DIR__, "generate_experiments.jl"))
    pop!(ARGS)
end
