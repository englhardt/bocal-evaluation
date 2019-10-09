config_file = isempty(ARGS) ? joinpath(@__DIR__, "config", "config_precompute_parameters.jl") : ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using SVDD, OneClassActiveLearning, Distributed, Gurobi, JuMP, MLKernels
using Serialization

include(joinpath(@__DIR__, "util", "setup_workers.jl"))

@info "Loading packages on all workers."
@everywhere using SVDD, OneClassActiveLearning, Gurobi, JuMP

@info "Searching for data sets in '$(data_input_root)'"
data_files = Vector{String}()
for data_set in vcat(values(data_dirs)...)
    data_dir = joinpath(data_input_root, data_set)
    if !isdir(data_dir)
        @info "Could not find dataset $data_dir"
        continue
    end
    local_names = readdir(data_dir)
    full_names = realpath.(joinpath.(data_input_root, data_dir, local_names))
    push!(data_files, full_names...)
end

@info "Initializing models."
all_model_params = @distributed vcat for file in data_files
    data, labels = load_data(file)
    n_obs = length(labels)
    model = instantiate(SVDDneg, data, fill(:U, n_obs), Dict{Symbol, Any}())
    initialize!(model, init_strategy)
    model_parameters = get_model_params(model)
    params = Dict{Symbol, Any}(
        :type => :SVDDneg,
        :file => file,
        :n_observations => n_obs,
        :C => get_model_params(model)[:C1],
        :gamma => MLKernels.getvalue(model.kernel_fct.alpha)
    )
    params
end

@info "Saving parameters."
model_params_by_filename = Dict{String, Dict{Symbol, Any}}()
for model_params in all_model_params
    model_params_by_filename[basename(model_params[:file])] = model_params
    delete!(model_params, :file)
end

serialize(data_parameter_file, model_params_by_filename)
@info "Done."
