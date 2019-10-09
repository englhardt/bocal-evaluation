using JuMP, Gurobi
include(joinpath(@__DIR__, "config.jl"))

experiment_name="baseline"

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

SAMPLE_LIMIT = 128
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

sequential_strategies = [:RandomPQs, :RandomOutlierPQs, :DecisionBoundaryPQs, :HighConfidencePQs, :NeighborhoodBasedPQs, :(BoundaryNeighborCombinationPQs{SVDD.SVDDneg})]
sequential_strategies = [Dict{Symbol, Any}(:type => s, :param => Dict{Symbol, Any}()) for s in sequential_strategies]

query_strategies = Vector{Dict{Symbol, Any}}()

for s in sequential_strategies
    push!(query_strategies, s)
end

for k in batch_sizes
    push!(query_strategies, Dict{Symbol, Any}(:type => :RandomBatchQs, :param => Dict{Symbol, Any}(:k => k)))
end

for k in batch_sizes
    for s in sequential_strategies
        cur_strat = Dict{Symbol, Any}(:type => :TopKBatchQs,
                                      :param => Dict{Symbol, Any}(:k => k, :SequentialStrategy => s))
        push!(query_strategies, cur_strat)
    end
end

for k in batch_sizes
    for s in sequential_strategies
        cur_strat = Dict{Symbol, Any}(:type => :GappedTopKBatchQs,
                                      :param => Dict{Symbol, Any}(:k => k, :m => k, :SequentialStrategy => s))
        push!(query_strategies, cur_strat)
    end
end
