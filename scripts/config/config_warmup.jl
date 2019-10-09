using JuMP, Gurobi
include(joinpath(@__DIR__, "config.jl"))

experiment_name="warmup"

data_dirs = ["Parkinson"]

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

SAMPLE_LIMIT = 2
batch_sizes = [2]

sequential_strategies = [:RandomPQs, :RandomOutlierPQs, :DecisionBoundaryPQs, :HighConfidencePQs, :NeighborhoodBasedPQs, :(BoundaryNeighborCombinationPQs{SVDD.SVDDneg})]

query_strategies = Vector{Dict{Symbol, Any}}()

for s in sequential_strategies
    push!(query_strategies, Dict{Symbol, Any}(:type => s, :param => Dict{Symbol, Any}()))
end

for k in batch_sizes
    push!(query_strategies, Dict{Symbol, Any}(:type => :RandomBatchQs, :param => Dict{Symbol, Any}(:k => k)))
end

sequential_strategy_types = [:DecisionBoundaryPQs]
sequential_strategies = [Dict{Symbol, Any}(:type => type, :param => Dict{Symbol, Any}()) for type in sequential_strategy_types]

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

rep_measures = [:KDE]
div_measures = [:AngleDiversity, :EuclideanDistance]
weight_combinations = [(1, 1, 1)]

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

rep_measures = [:KDE]
div_measures = [:EuclideanDistance]

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
