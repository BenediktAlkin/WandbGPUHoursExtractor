from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
import wandb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="https://api.wandb.ai/")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument(
        "--world_size_config", 
        type=str, 
        default="dist/world_size",
        help="field in the W&B config where the world_size is stored (run.config[world_size_config])",
    )
    parser.add_argument(
        "--default_world_size",
        type=int, 
        default=0,
        help="default world_size which is used when a run doesn't have the world_size_config field",
    )
    return vars(parser.parse_args())

def main(host, entity, project, world_size_config, default_world_size):
    print(f"host: {host}")
    print(f"entity: {entity}")
    print(f"project: {project}")
    print(f"world_size_config: {world_size_config}")
    print(f"default_world_size: {default_world_size}")
    wandb.login(host=host)
    api = wandb.Api()
    data = []
    for i, run in enumerate(api.runs(f"{entity}/{project}")):
        if run.state == "running":
            continue
        data.append(
            dict(
                start=datetime.strptime(run.createdAt, "%Y-%m-%dT%H:%M:%S"),
                end=datetime.strptime(run.heartbeatAt, "%Y-%m-%dT%H:%M:%S"),
                world_size=run.config.get(world_size_config, default_world_size),
            )
        )
    wandb.finish()

    data = pd.DataFrame(data)
    total = ((data["end"] - data["start"]) * data["world_size"]).sum()
    total_hours = total.total_seconds() / 3600
    start = data["start"].min()
    end = data["end"].max()
    print(f"statistics from {start:%Y-%m-%d} to {end:%Y-%m-%d} ({(end - start).total_seconds() / 86400:.0f} days)")
    print(f"num_runs: {len(data)}")
    print(f"total: {total.days} days {total.seconds / 3600:.0f} hours")
    print(f"GPU-hours: {total_hours:.0f}")
    print(f"GPU-hours per run: {total_hours / len(data):.2f}")


if __name__ == "__main__":
    main(**parse_args())
