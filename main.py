from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
import wandb


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="https://api.wandb.ai/")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    return vars(parser.parse_args())


def get_world_size(run):
    return run.config.get("dist/world_size", 0)


def main(host, entity, project):
    print(f"host: {host}")
    print(f"entity: {entity}")
    print(f"project: {project}")
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
                world_size=get_world_size(run),
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
