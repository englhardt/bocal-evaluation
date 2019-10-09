using JuMP, Gurobi
include(joinpath(@__DIR__, "config.jl"))

experiment_name="iterative"

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

SAMPLE_LIMIT = 128
batch_sizes = [2, 4, 8, 16]

sequential_strategy_types = [:DecisionBoundaryPQs, :HighConfidencePQs]
sequential_strategies = [Dict{Symbol, Any}(:type => type, :param => Dict{Symbol, Any}()) for type in sequential_strategy_types]

rep_measures = [:KDE]
div_measures = [:AngleDiversity, :EuclideanDistance]

relative_importances = [0, 1, 2, 4, 8]

weight_combinations = [
    (λ_inf, λ_rep, λ_div)
    for λ_div in relative_importances
    for λ_rep in relative_importances
    for λ_inf in relative_importances
]
weight_combinations = map(x -> x ./ gcd(x...), weight_combinations) |> unique
filter!(x -> sum(x) > 0, weight_combinations)

query_strategies = Vector{Dict{Symbol, Any}}()

for k in batch_sizes
    for s in sequential_strategies
        for r in rep_measures
            for d in div_measures
                for (λ_inf, λ_rep, λ_div) in weight_combinations
                    cur_strat = Dict{Symbol, Any}(
                        :type => :IterativeBatchQs,
                        :param => Dict{Symbol, Any}(
                            :k => k,
                            :SequentialStrategy => s,
                            :representativeness => r,
                            :diversity => d,
                            :λ_inf => λ_inf,
                            :λ_rep => λ_rep,
                            :λ_div => λ_div))
                    push!(query_strategies, cur_strat)
                end
            end
        end
    end
end
