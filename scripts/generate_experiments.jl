isempty(ARGS) && error("Please pass a config file as command line argument.")
config_file = ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using SVDD, OneClassActiveLearning
using JuMP
using Gurobi
using MLKernels
using Memento
using Random
using Serialization

function setup_experiment_folder(config_file)
    isdir(exp_dir) || mkpath(exp_dir)
    mkpath(joinpath(exp_dir, "log", "experiment"))
    mkpath(joinpath(exp_dir, "log", "worker"))
    mkpath(joinpath(exp_dir, "results"))
    cp(config_file, joinpath(exp_dir, basename(config_file)))
    cp(joinpath(first(splitdir(config_file)), "config.jl"), joinpath(exp_dir, "config.jl"))
end

function create_experiments(query_strategies::Vector{Dict{Symbol, Any}}, SAMPLE_LIMIT::Int, solver::JuMP.OptimizerFactory)::Vector{Dict{Symbol, Any}}
    experiment_configurations = [
        (data_file, model_params_by_filename[basename(data_file)], query_strategy, SAMPLE_LIMIT)
        for query_strategy in query_strategies
        for data_file in data_files
    ]
    all_experiments = []
    for (data_file, model_params, query_strategy, SAMPLE_LIMIT) in experiment_configurations
        num_samples = model_params[:n_observations]

        data_set_name = splitdir(splitdir(data_file)[1])[2]
        local_data_file_name = splitext(basename(data_file))[1]

        output_path = joinpath(exp_dir, "results", data_set_name)
        isdir(output_path) || mkpath(output_path)

        batch_size = haskey(query_strategy[:param], :k) ? query_strategy[:param][:k] : 1
        al_iterations = ceil(Int, min(num_samples, SAMPLE_LIMIT) / batch_size)

        experiment = Dict{Symbol, Any}(
            :data_file => data_file,
            :data_set_name => data_set_name,
            :log_dir => joinpath(exp_dir, "log"),
            :split_strategy_name => "Sf",
            :initial_pool_strategy_name => "Pu",
            :model => Dict(:type => :SVDDneg,
                           :param => Dict{Symbol, Any}(),
                           :init_strategy => SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(model_params[:gamma])), FixedCStrategy(model_params[:C]))),
            :query_strategy => query_strategy,
            :oracle => Dict{Symbol, Any}(
                :type => :PoolOracle,
                :param => Dict{Symbol, Any}()
            ),
            :split_strategy => OneClassActiveLearning.DataSplits(trues(num_samples)),
            :param => Dict{Symbol, Any}(:num_al_iterations => al_iterations,
                                        :initial_pools => fill(:U, num_samples),
                                        :solver => solver,
                                        :adjust_K => true,
                                        :initial_pool_resample_version => 1
                                        )
        )
        exp_hash = hash(experiment)
        experiment[:hash] = "$(exp_hash)"
        experiment[:output_file] = joinpath(output_path, "$(local_data_file_name)_$(query_strategy[:type])_SVDDneg_$(exp_hash).json")
        push!(all_experiments, experiment)
    end
    @info "Created $(realpath(exp_dir)) with $(length(all_experiments)) experiment settings."
    return all_experiments
end

function save_experiments(all_experiments::Array{Dict{Symbol, Any},1})
    exp_hashes_file = joinpath(exp_dir, "experiment_hashes")
    @info "Saving experiment hashes at $exp_hashes_file..."
    open(exp_hashes_file, "a") do f
        for e in all_experiments
            write(f, "$(e[:hash])\n")
        end
    end
    @info "... done."
    exp_filename = joinpath(exp_dir, "experiments.jser")
    serialize(exp_filename, all_experiments)
    @info "Experiments written to file $(exp_filename)."
end

# setup experiment directory
exp_dir = joinpath(data_output_root, experiment_name)
if isdir(exp_dir)
    print("Type 'yes' or 'y' to delete and overwrite experiment $(exp_dir): ")
    argin = readline()
    if argin == "yes" || argin == "y"
        rm(exp_dir, recursive=true)
    else
        error("Aborting...")
    end
end
mkpath(exp_dir)
exp_dir = realpath(exp_dir)

# find all data files
all(isdir.(joinpath.(data_input_root, data_dirs))) || error("Not all data dirs are valid.")
data_files = vcat(([joinpath.(data_input_root, x, readdir(joinpath(data_input_root, x))) for x in data_dirs])...)
@info "Found $(length(data_files)) data files."

# parse precompute parameters
isfile(data_parameter_file) || error("Precomputed parameters not found. Please follow the instructions in the Readme.")
@info "Using precompute parameters from file \"$data_parameter_file\"."
model_params_by_filename = deserialize(data_parameter_file)

# generate experiments
setup_experiment_folder(config_file)
Random.seed!(0)
experiments = create_experiments(query_strategies, SAMPLE_LIMIT, solver)
save_experiments(experiments)
