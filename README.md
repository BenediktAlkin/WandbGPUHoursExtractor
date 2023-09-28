# WandbGPUHoursExtractor

Extracts GPU-hours statistics of a [Weights & Biases](https://wandb.ai) project using the [Import & Export API](https://docs.wandb.ai/ref/python/public-api/).

# Usage
```
python main.py \
    --entity <ENTITY> \
    --project <PROJECT> \
    --world_size_config <WORLD_SIZE_CONFIG> \
    --default_world_size <DEFAULT_WORLD_SIZE> \
    --startdate <DEFAULT_WORLD_SIZE> \
    --enddate <DEFAULT_WORLD_SIZE>
```
For self-hosted W&B servers the `--host <HOST>` flag is required (by default the official W&B server is used).

As W&B can't track how many GPUs are used for a run (the so-called "world size"), you have to track this yourself via a field in the W&B config:
```
wandb.init(...)
wandb.config["dist/world_size"] = 2
wandb.finish(...)
```

If a run doesn't have this flag, you can specify a default value via `--default_world_size <DEFAULT_WORLD_SIZE>`.
The default value is 0, which discards all runs that don't have the `world_size_config` flag set.
If all your runs use only a single GPU, you can use `--default_world_size 1` instead of specifying `--world_size_config`.

You can specify a start/end date via `--startdate <YYYY-MM-DD>` and `--enddate <YYYY-MM-DD>`. 
For example to start from 15th january 2023: `--startdate 2023-01-15`.
These filters will always look at the time when a run **finished, not when it started** 
(e.g. with `--startdate 2023-01-15 --enddate 2023-02-15` all runs that have finished between 15th january and 
15th february will be counted towards the total number of GPU-hours). If you want statistics over multiple timespans, make sure to 
use the exact enddate of a previous timespan as the startdate of the next timespan 
(e.g. you want statistics from 15th of january to 15th of march but split into months -> run once with
`--startdate 2023-01-15 --enddate 2023-02-15` and once with `--startdate 2023-02-15 --enddate 2023-03-15`) 

Runs that are still in progress (state == "running") are excluded.
However, sometimes runs are stuck in the "running" state, despite them being already finished.
These runs will be counted towards the total if they ended more than 1 day ago 
(the end time of a run that is stuck on running is equal to its "Created At" timestamp + "Runtime").
 

## Example output
```
host: https://api.wandb.ai/
entity: <ENTITY>
project: <PROJECT>
world_size_config: dist/world_size
default_world_size: 0
statistics from 2023-05-13 to 2023-06-26 (44 days)
num_runs: 842
total: 679 days 11 hours
GPU-hours: 16307
GPU-hours per run: 19.37
```

## Self-hosted W&B Server
Specify the hostname of a self-hosted W&B server via `--host HOSTNAME`.
