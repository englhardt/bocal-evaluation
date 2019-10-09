config_file = isempty(ARGS) ? joinpath(@__DIR__, "config", "config.jl") : ARGS[1]
isfile(config_file) || error("Cannot read '$config_file'")
println("Config supplied: '$config_file'")
include(config_file)

using OneClassActiveLearning, SVDD
using JSON, Unmarshal, DataFrames, CSV
using Distributed

function process_exp_dir(exp_dir, exp_scenario, output_dir, target_metric=:matthews_corr)
    @info "Looking for experiments at $(exp_dir)"
    all_json_files = []
    for data_set in readdir(exp_dir)
        if endswith(data_set, ".csv")
            continue
        end
        json_results = filter(x -> endswith(x, "json"), readdir(joinpath(exp_dir, data_set)))
        json_files = joinpath.(exp_dir, data_set, json_results)
        @info "Found $(length(json_files)) experiments on data set $(data_set)"
        append!(all_json_files, json_files)
    end


    @info "Creating csv data from $(length(all_json_files)) results..."
    all_data = @distributed vcat for json_file in all_json_files
        @info "Processing '$json_file'."
        res = Unmarshal.unmarshal(OneClassActiveLearning.Result, JSON.parsefile(json_file))
        query_strategy = missing
        if res.experiment[:query_strategy][:type] isa String
            query_strategy = res.experiment[:query_strategy][:type]
        else
            query_strategy = String(res.experiment[:query_strategy][:type][:args][1])
        end
        data_set_name = res.experiment[:data_set_name]
        params = res.experiment[:query_strategy][:param]
        batch_size = get(params, :k, 1.0)

        if haskey(res.al_history, target_metric)
            scores = values(res.al_history, target_metric)

            cms = values(res.al_history, :cm)
            num_inliers = cms[1].tn + cms[1].fp
            num_outliers = cms[1].tp + cms[1].fn
            tp_rate = [x.tp / num_outliers for x in cms]
            fn_rate = [x.fn / num_outliers for x in cms]
            fp_rate = [x.fp / num_inliers for x in cms]
            tn_rate = [x.tn / num_inliers for x in cms]

            angle_batch_diversity = vcat(0, values(res.al_history, :angle_batch_diversity))
            euclidean_batch_diversity = vcat(0, values(res.al_history, :euclidean_batch_diversity))

            num_observations = res.data_stats.num_observations
            num_dimensions = res.data_stats.num_dimensions
            num_iterations = length(cms)
            num_labeled_samples = min.(Array(0:(length(scores)-1)) * batch_size, num_observations)

            time_qs = vcat(0, values(res.al_history, :time_qs))
            time_fit = values(res.al_history, :time_fit)

        else
            scores, tp_rate, fn_rate, fp_rate, tn_rate, time_qs, time_fit = fill(missing, 7)
            angle_batch_diversity, euclidean_batch_diversity = fill(missing, 2)
            num_observations, num_dimensions, num_labeled_samples = fill(missing, 3)
        end

        sequential_strategy = missing
        if haskey(params, :SequentialStrategy)
            if params[:SequentialStrategy][:type] isa String
                sequential_strategy = String(params[:SequentialStrategy][:type])
            else
                sequential_strategy = String(params[:SequentialStrategy][:type][:args][1])
            end
        end
        representativeness = get(params, :representativeness, missing)
        diversity = get(params, :diversity, missing)
        l_inf = get(params, :λ_inf, missing)
        l_rep = get(params, :λ_rep, missing)
        l_div = get(params, :λ_div, missing)
        result_data = DataFrame(
            id = res.id,
            data_set_name = data_set_name,
            query_strategy = query_strategy,
            k = batch_size,
            num_observations = num_observations,
            num_dimensions = num_dimensions,
            labeled_samples = num_labeled_samples,
            scores = scores,
            tp_rate = tp_rate,
            tn_rate = tn_rate,
            fp_rate = fp_rate,
            fn_rate = fn_rate,
            informativeness = sequential_strategy,
            representativeness = representativeness,
            diversity = diversity,
            l_inf = l_inf,
            l_rep = l_rep,
            l_div = l_div,
            angle_batch_diversity = angle_batch_diversity,
            euclidean_batch_diversity = euclidean_batch_diversity,
            time_qs = time_qs,
            time_fit = time_fit
        )
        if haskey(res.al_history, target_metric)
            scores_summary = res.al_summary[target_metric]
            start_quality = scores_summary[:start_quality]
            end_quality = scores_summary[:end_quality]
            average_gain = scores_summary[:average_gain]
            average_loss = scores_summary[:average_loss]
            maximum = scores_summary[:maximum]
            average_quality_change = scores_summary[:average_quality_change]
            ratio_of_outlier_queries = scores_summary[:ratio_of_outlier_queries]
            idx_16_lab = round(Int, 16 / batch_size + 1)
            initial_gain = length(scores) >= idx_16_lab ? scores[idx_16_lab] - scores[1] : missing
            total_angle_batch_diversity = sum(angle_batch_diversity)
            total_euclidean_batch_diversity = sum(euclidean_batch_diversity)
            total_time_fit = sum(time_fit)
            total_time_qs = sum(time_qs)
            time_exp = res.al_summary[:runtime][:time_exp]
        else
            start_quality, end_quality, average_gain, average_loss, maximum, average_quality_change,
                ratio_of_outlier_queries, initial_gain, total_time_fit, total_time_qs, time_exp,
                total_angle_batch_diversity, total_euclidean_batch_diversity = fill(missing, 13)
        end
        summary_data = DataFrame(
            id = res.id,
            data_set_name = data_set_name,
            query_strategy = query_strategy,
            k = batch_size,
            informativeness = sequential_strategy,
            representativeness = representativeness,
            diversity = diversity,
            l_inf = l_inf,
            l_rep = l_rep,
            l_div = l_div,
            start_quality = start_quality,
            end_quality = end_quality,
            average_gain = average_gain,
            average_loss = average_loss,
            initial_gain = initial_gain,
            maximum = maximum,
            average_quality_change = average_quality_change,
            ratio_of_outlier_queries = ratio_of_outlier_queries,
            angle_batch_diversity = total_angle_batch_diversity,
            euclidean_batch_diversity = total_euclidean_batch_diversity,
            time_fit = total_time_fit,
            time_qs = total_time_qs,
            time_exp = time_exp,
            exit_code = string(res.status[:exit_code])
        )
        (result_data, summary_data)
    end

    all_result_data = vcat(map(first, all_data)...)
    all_summary_data = vcat(map(last, all_data)...)
    @info "... done."

    data_file = joinpath(output_dir, "result_data_$(exp_scenario).csv")
    @info "Writing data to $(data_file)..."
    CSV.write(data_file, all_result_data)
    @info "... done."
    summary_file = joinpath(output_dir, "summary_data_$(exp_scenario).csv")
    @info "Writing data to $(summary_file)..."
    CSV.write(summary_file, all_summary_data)
    @info "... done."
end

include(joinpath(@__DIR__, "util", "setup_workers.jl"))
@info "Loading packages on all workers."
@everywhere using Pkg, DataFrames, JSON, Unmarshal, OneClassActiveLearning

for e in [x for x in readdir(data_output_root) if !occursin(".csv", x) && !occursin("warmup", x)]
    exp_dir = realpath(joinpath(data_output_root, e, "results"))
    @info "Processing '$exp_dir'."
    process_exp_dir(exp_dir, e, data_output_root)
end

rmprocs(workers())
