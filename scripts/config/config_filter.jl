using JuMP, Gurobi
include(joinpath(@__DIR__, "config.jl"))

experiment_name="filter"

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

SAMPLE_LIMIT = 128
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

sequential_strategy_types = [:DecisionBoundaryPQs, :HighConfidencePQs, :NeighborhoodBasedPQs]
sequential_strategies = [Dict{Symbol, Any}(:type => type, :param => Dict{Symbol, Any}()) for type in sequential_strategy_types]

rep_measures = [:KDE]
div_measures = [:AngleDiversity, :EuclideanDistance]

query_strategies = Vector{Dict{Symbol, Any}}()

for k in batch_sizes
    for s in sequential_strategies
        for div in div_measures
            cur_strat = Dict{Symbol, Any}(:type => :FilterSimilarBatchQs,
                                          :param => Dict{Symbol, Any}(
                                            :k => k,
                                            :SequentialStrategy => s,
                                            :diversity => div))
            push!(query_strategies, cur_strat)
        end
    end
end

for k in batch_sizes
    for s in sequential_strategies
        for r in rep_measures
            for d in div_measures
                cur_strat = Dict{Symbol, Any}(
                    :type => :FilterHierarchicalBatchQs,
                    :param => Dict{Symbol, Any}(
                        :k => k,
                        :SequentialStrategy => s,
                        :representativeness => r,
                        :diversity => d))
                push!(query_strategies, cur_strat)
            end
        end
    end
end
