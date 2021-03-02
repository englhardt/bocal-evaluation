# Batch One-Class Active Learning Evaluation
_Scripts and notebooks to benchmark batch one-class active learning strategies._

This repository contains scripts and notebooks to reproduce the experiments and analyses of the paper

> Adrian Englhardt, Holger Trittenbach, Dennis Vetter, Klemens Böhm, “Finding the Sweet Spot: Batch Selection for One-Class Active Learning”. In: Proceedings of the 2020 SIAM International Conference on Data Mining (SDM), DOI: [10.1137/1.9781611976236.14](https://doi.org/10.1137/1.9781611976236.14), May 7-9, 2020, Cincinnati, Ohio, USA.

For more information about this research project, see also the [BOCAL project](https://www.ipd.kit.edu/bocal/) website.
For a general overview and a benchmark on one-class active learning see the [OCAL project website](https://www.ipd.kit.edu/ocal/).

The analysis and main results of the experiments can be found under [notebooks](https://github.com/englhardt/bocal-evaluation/tree/master/notebooks):
  * `batch_criteria_example.ipynb`: Figure 1
  * `evaluation.ipynb`: Figure 2, Table 1 and Table 2

To execute the notebooks, make sure you follow the [setup](#setup), and download the [raw results](https://www.ipd.kit.edu/bocal/output.zip) into `data/output/`.

## Prerequisites

The experiments are implemented in [Julia](https://julialang.org/), some of the evaluation notebooks are written in python.
This repository contains code to setup the experiments, to execute them, and to analyze the results.
The one-class classifiers and active learning methods are implemented in two separate Julia packages: [SVDD.jl](https://github.com/englhardt/SVDD.jl) and [OneClassActiveLearning.jl](https://github.com/englhardt/OneClassActiveLearning.jl).

### Setup

Just clone the repo.
```bash
$ git clone https://github.com/englhardt/bocal-evaluation.git
```
* Experiments require Julia 1.1.1, requirements are defined in `Manifest.toml`. To instantiate, start julia in the `bocal-evaluation` directory with `julia --project` and run `julia> ]instantiate`. See [Julia documentation](https://docs.julialang.org/en/v1.0/stdlib/Pkg/#Using-someone-else's-project-1) for general information on how to setup this project.
* Notebooks require
  * Julia 1.1.1 (dependencies are already installed in the previous step)
  * Python 3.7 and `pipenv`. Run `pipenv install` to install all dependencies

## Repo Overview

* `data`
  * `input`
    * `raw`: contains unprocessed data set collections `literature` and `semantic` from the [DAMI](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/) repository
    * `processed`: output directory of _preprocess_data.jl_
  * `output`: output directory of experiments; _generate_all_experiments.jl_ creates the folder structure and experiments; _run_batch_experiments.jl_ writes results and log files
* `notebooks`: jupyter notebooks to analyze experimental results
  * `batch_criteria_example.ipynb`: Figure 1
  * `evaluation.ipynb`: Figure 2, Table 1 and Table 2
* `scripts`
  * `config`: configuration files for experiments
    * `config.jl`: high-level configuration, e.g., for number of workers
    * `config_baseline.jl`: experiment config for baseline batch strategies
    * `config_filter.jl`: experiment config for filter batch strategies
    * `config_iterative.jl`: experiment config for the iterative batch strategy
    * `config_partition.jl`: experiment config for partitioning batch strategies
    * `config_precompute_parameters.jl`: experiment config to precompute classifier hyperparameters
    * `config_warmup.jl`: experiment config for precompulation warmup experiments
  * `util/setup_workers.jl`: utility script to setup multiple workers, see LINK
  * `generate_all_experiments.jl`: generate all experiments
  * `generate_experiments.jl`: generate experiments for one type of query strategy, e.g. baseline
  * `precompute_parameters.jl`: precompute classifier hyperparameters
  * `preprocess_data.jl`: preprocess DAMI data sets
  * `reduce_results.jl`: combine result files into a several CSV files.
  * `run_batch_experiments.jl`: executes experiments

## Experiment Pipeline

The experiment pipeline uses config files to set paths and experiment parameters.
There are two types of config files:
* `scripts/config.jl`: this config defines high-level information on the experiment, such as number of workers, where the data files are located, and log levels.
* `scripts/<config_baseline|...>.jl`: These config files define the experimental grid, including the data sets, classifiers, and active-learning strategies.

1. _Data Preprocessing_: The preprocessing step transforms publicly available benchmark data sets into a common csv format, and performs feature selection.
  * **Input:** Download [semantic.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/semantic.tar.gz) and [literature.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/literature.tar.gz) containing the .arff files from the DAMI benchmark repository and extract into `data/input/raw/.../<data set>` (e.g. `data/input/raw/literature/ALOI/` or `data/input/raw/semantic/Annthyroid`).
  * **Execution:**
  ```bash
     $ julia --project scripts/preprocess_data.jl
  ```
  * **Output:** .csv files in `data/input/processed/`

  We also provide our preprocessed data to [download](https://www.ipd.kit.edu/bocal/input.zip) (8.4 MB).

2. _Precompute Paramters_: This step precomputes the hyperparameters of the classifier for each data set.
  * **Input:** Preprocessed data files
  * **Execution:**
  ```bash
     $ julia --project scripts/precompute_parameters.jl
  ```
  * **Output:** Precompute parameters stored in `data/input/processed/parameters.jser`

3. _Generate Experiments_: This step creates a set of experiments. For the synthetic evaluation the scripts generate the data as well.
  * **Input**: Preprocessed data files
  * **Execution:**
  ```bash
     $ julia --project scripts/generate_all_experiments.jl
  ```
  * **Output:**
    * Creates an experiment directory with the naming `<exp_name>`. The directories created contains several items:
      * `log` directory: skeleton for experiment logs (one file per experiment), and worker logs (one file per worker)
      * `results` directory: skeleton for result files
      * `experiments.jser`: this contains a serialized Julia Array with experiments. Each experiment is a Dict that contains the specific combination. Each experiment can be identified by a unique hash value.
      * `experiment_hashes`: file that contains the hash values of the experiments stored in `experiments.jser`
      * `config.jl` and `config_<scenario>.jl`: a copy of the config file used to generate the experiments

4. _Run Experiments_: This step executes the experiments created in Step 2.
Each experiment is executed on a worker. In the default configuration, a worker is one process on the localhost.
For distributed workers, see Section [Infrastructure and Parallelization](#infrastructure-and-parallelization).
A worker takes one specific configuration, runs the active learning experiment, and writes result and log files.
  * **Input:** Generated experiments from step 2
  * **Execution:**
  ```bash
     $ julia --project scripts/run_batch_experiments.jl
  ```
  * **Output:** The output files are named by the experiment hash and are .json files (e.g., `data/output/baseline/results/ALOI/ALOI_withoutdupl_norm_r01_DecisionBoundaryPQs_SVDDneg_6935936306455490995.json`)

5. _Reduce Results_: Merge experimental results into .csv files by using summary statistics
  * **Input:** Full path to finished experiments.
  * **Execution:**
  ```bash
     $ julia --project scripts/reduce_results.jl
  ```
  * **Output:** Multiple result csv files for each scenario, e.g., `data/output/summary_data_baseline.csv`.

6. _Analyze Results:_ jupyter notebooks in the `notebooks`directory to analyze the reduced `.csv`. Run the following to produce the figures and tables in the experiment section of the paper:
  ```bash
    $ pipenv run evaluation
  ```

## Infrastructure and Parallelization

Step 4 _Run Experiments_ can be parallelized over several workers. In general, one can use any [ClusterManager](https://github.com/JuliaParallel/ClusterManagers.jl). In this case, the node that executes `run_experiments.jl` is the driver node. The driver node loads the `experiments.jser`, and initiates a function call for each experiment on one of the workers via `pmap`.

## Authors
This package is developed and maintained by [Adrian Englhardt](https://github.com/englhardt/)
