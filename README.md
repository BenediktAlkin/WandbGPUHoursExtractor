# WandbGPUHoursExtractor

Extracts GPU-hours statistics of a [Weights & Biases](https://wandb.ai) project using the [Import & Export API](https://docs.wandb.ai/ref/python/public-api/).

# Usage
`python main.py --entity <ENTITY> --project <PROJECT>`

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

## Self-hosted W&B Server
Specify the hostname of a self-hosted W&B server via `--host HOSTNAME`.

`python main.py --host https://api.wandb.ai/ --entity <ENTITY> --project <PROJECT>`