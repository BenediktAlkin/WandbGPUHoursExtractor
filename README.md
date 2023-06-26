# WandbGPUHoursExtractor

Extracts GPU-hours statistics of a [Weights & Biases](https://wandb.ai) project using the [Import & Export API](https://docs.wandb.ai/ref/python/public-api/).

# Usage
```
python main.py \
    --entity <ENTITY> \
    --project <PROJECT> \
    --world_size_config <WORLD_SIZE_CONFIG> \
    --default_world_size <DEFAULT_WORLD_SIZE>
```

As W&B can't track how many GPUs are used for a run (the so-called "world size"), you have to track this yourself via a field in the W&B config:
```
wandb.init(...)
wandb.config["dist/world_size"] = 2
wandb.finish(...)
```

If a run doesn't have this flag, you can specify a default value via `--default_world_size <DEFAULT_WORLD_SIZE>`.
The default value is 0, which discards all runs that don't have the `world_size_config` flag set.
If all your runs use only a single GPU you can use `--default_world_size 1` instead of specifying `--world_size_config`.


## Example output
```
host: https://api.wandb.ai/
entity: <ENTITY>
project: <PROJECT>
statistics from 2023-05-13 to 2023-06-26 (44 days)
num_runs: 842
total: 679 days 11 hours
GPU-hours: 16307
GPU-hours per run: 19.37
```

NOTE: as W&B does not track how many GPUs are used you have to set this yourself via a field in the config

## Self-hosted W&B Server
Specify the hostname of a self-hosted W&B server via `--host HOSTNAME`.

`python main.py --host https://api.wandb.ai/ --entity <ENTITY> --project <PROJECT>`