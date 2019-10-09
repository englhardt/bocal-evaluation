config_file = isempty(ARGS) ? joinpath(@__DIR__, "config", "config.jl") : ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using SVDD, OneClassActiveLearning
using Memento, Gurobi, JSON, JuMP
using Random
using Distributed
using MLKernels
using Serialization
using Dates

@info "Running experiment in \"$(realpath(data_output_root))\"."
include(joinpath(@__DIR__, "util", "setup_workers.jl"))

@info "Loading packages on all workers."
@everywhere using Pkg
@everywhere using SVDD, OneClassActiveLearning, JSON, JuMP, Gurobi, Memento, Random, MLKernels, Serialization
@everywhere import SVDD: SVDDneg
@info "Loaded."

@everywhere fmt_string = "[{name} | {date} | {level}]: {msg}"
@everywhere loglevel = "debug"

@everywhere function setup_logging(experiment)
    setlevel!(getlogger("root"), "error")
    setlevel!(getlogger(OneClassActiveLearning), loglevel)
    setlevel!(getlogger(SVDD), loglevel)

    exp_logfile = joinpath(experiment[:log_dir], "experiment", "$(experiment[:hash]).log")
    worker_logfile = joinpath(experiment[:log_dir], "worker", "$(gethostname())_$(getpid()).log")

    WORKER_LOGGER = Memento.config!("runner", "debug"; fmt=fmt_string)

    exp_handler = DefaultHandler(exp_logfile, DefaultFormatter(fmt_string))
    push!(getlogger(OneClassActiveLearning), exp_handler, experiment[:hash])
    push!(getlogger(SVDD), exp_handler, experiment[:hash])
    push!(WORKER_LOGGER, exp_handler, experiment[:hash])

    worker_handler = DefaultHandler(worker_logfile, DefaultFormatter(fmt_string))
    setlevel!(gethandlers(WORKER_LOGGER)["console"], "error")
    push!(WORKER_LOGGER, worker_handler)

    return WORKER_LOGGER
end

@everywhere function cleanup_logging(worker_logger::Logger, experiment_hash)
    delete!(gethandlers(getlogger(OneClassActiveLearning)), experiment_hash)
    delete!(gethandlers(getlogger(SVDD)), experiment_hash)
    delete!(gethandlers(worker_logger), experiment_hash)
    return nothing
end

@everywhere function Memento.warn(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::ErrorException)
    Memento.warn(logger, "Caught ErrorException, msg='$(error.msg)')")
end

@everywhere function run_experiment(experiment::Dict, warmup=false)
    # Make experiments deterministic
    Random.seed!(0)

    WORKER_LOGGER = setup_logging(experiment)
    info(WORKER_LOGGER, "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash])")
    if isfile(experiment[:output_file])
        warn(WORKER_LOGGER, "Aborting experiment because the output file already exists. Filename: $(experiment[:output_file])")
        cleanup_logging(WORKER_LOGGER, experiment[:hash])
        return nothing
    end

    res = Result(experiment)
    errorfile = joinpath(experiment[:log_dir], "worker", "$(gethostname())_$(getpid())")
    try
        time_exp = @elapsed res = OneClassActiveLearning.active_learn(experiment)
        res.al_summary[:runtime] = Dict(:time_exp => time_exp)
    catch e
        res.status[:exit_code] = Symbol(typeof(e))
        @warn "Experiment $(experiment[:hash]) finished with unkown error."
        @warn e
        @warn stacktrace(catch_backtrace())
        warn(WORKER_LOGGER, "Experiment $(experiment[:hash]) finished with unkown error.")
        warn(WORKER_LOGGER, e)
    finally
        if res.status[:exit_code] != :success
            info(WORKER_LOGGER, "Writing error hash to $errorfile.error.")
            open("$errorfile.error", "a") do f
                print(f, "$(experiment[:hash])\n")
            end
        end
        info(WORKER_LOGGER, "Writing result to $(experiment[:output_file]).")
        !WARMUP && OneClassActiveLearning.write_result_to_file(experiment[:output_file], res)
        cleanup_logging(WORKER_LOGGER, experiment[:hash])
    end

    return nothing
end

@info "Warming up workers."
try
    @everywhere begin
        WARMUP = true
        include($config_file)
        warmup_experiments = deserialize(open(joinpath(data_output_root, "warmup", "experiments.jser")))
        print("Starting warmup.")
        map(run_experiment, warmup_experiments)
        print("Finished warmup.")
    end
catch e
    @warn e
    rmprocs(workers())
    @info "Warmup failed. Terminating."
    exit()
end
@info "Warmup done."

@everywhere WARMUP = false
# load and run experiments
all_experiments = []
for s in readdir(data_output_root)
    if occursin(".csv", s) || occursin("warmup", s)
        @info "skipping $s"
        continue
    end
    @info "Running experiments in directory $s"
    exp_dir = joinpath(data_output_root, s)
    @info "Loading experiments.jser"
    experiments = deserialize(open(joinpath(exp_dir, "experiments.jser")))
    append!(all_experiments, experiments)
end

try
    @info "Running $(length(all_experiments)) experiments."
    start = now()
    @info "$(start) - Running experiments..."
    pmap(run_experiment, all_experiments, on_error=ex->print("!!! ", ex))
    finish = now()
    @info "$finish - Done."
    @info "Ran $(length(all_experiments)) experiment(s) in $(canonicalize(Dates.CompoundPeriod(finish - start)))"
catch e
    @warn e
finally
    rmprocs(workers())
end
