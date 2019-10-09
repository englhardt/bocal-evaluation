JULIA_ENV = realpath(joinpath(@__DIR__, "..", ".."))
data_root = realpath(joinpath(@__DIR__, "..", "..", "data"))
data_raw_input_root = joinpath(data_root, "input", "raw")
data_input_root = joinpath(data_root, "input", "processed")
data_parameter_file = joinpath(data_input_root, "parameters.jser")
data_output_root = joinpath(data_root, "output")

worker_list = [("localhost", 1)]
exeflags = `--project="$JULIA_ENV"`
sshflags= `-i path/to/ssh/key/file`

fmt_string = "[{name} | {date} | {level}]: {msg}"
loglevel = "debug"

### data directories ###
raw_data_dirs = Dict("semantic" => ["Arrhythmia", "Annthyroid", "Cardiotocography", "HeartDisease", "Hepatitis", "PageBlocks", "Parkinson", "Pima", "SpamBase", "Stamps"],
                     "literature" => ["ALOI", "Glass", "Ionosphere", "KDDCup99", "Lymphography", "PenDigits", "Shuttle", "WBC", "WDBC", "WPBC", "Waveform"])
data_dirs = vcat(values(raw_data_dirs)...)
