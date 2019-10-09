include("config.jl")

using JuMP, Gurobi, SVDD

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)
init_strategy = WangCombinedInitializationStrategy(solver, BoundedTaxErrorEstimate(0.05, 0.02, 0.98))
