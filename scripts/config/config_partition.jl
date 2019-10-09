using JuMP, Gurobi
include(joinpath(@__DIR__, "config.jl"))

experiment_name="partition"

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

SAMPLE_LIMIT = 128
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

sequential_strategy_types = [:DecisionBoundaryPQs, :HighConfidencePQs, :NeighborhoodBasedPQs]
sequential_strategies = [Dict{Symbol, Any}(:type => type, :param => Dict{Symbol, Any}()) for type in sequential_strategy_types]

query_strategies = Vector{Dict{Symbol, Any}}()

for k in batch_sizes
    push!(query_strategies, Dict{Symbol, Any}(:type => :ClusterBatchQs, :param => Dict{Symbol, Any}(:k => k)))
end

for k in batch_sizes
    for s in sequential_strategies
        cur_strat = Dict{Symbol, Any}(:type => :ClusterTopKBatchQs,
                                      :param => Dict{Symbol, Any}(:k => k, :SequentialStrategy => s))
        push!(query_strategies, cur_strat)
    end
end

for k in batch_sizes
    for s in sequential_strategies
        cur_strat = Dict{Symbol, Any}(:type => :EnsembleBatchQs,
                                      :param => Dict{Symbol, Any}(:k => k, :SequentialStrategy => s, :solver => solver))
        push!(query_strategies, cur_strat)
    end
end
