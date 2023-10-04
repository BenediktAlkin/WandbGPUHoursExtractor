from argparse import ArgumentParser
from datetime import datetime, timedelta

import pandas as pd
import wandb
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--host", type=str, default="https://api.wandb.ai/")
    parser.add_argument(
        "--startdate",
        type=str,
        help="date to start counting GPU-hours in YYYY-MM-DD format (e.g. 2023-01-15)",
    )
    parser.add_argument(
        "--enddate",
        type=str,
        help="date to end counting GPU-hours in YYYY-MM-DD format (e.g. 2023-02-15)",
    )
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
    parser.add_argument(
        "--accelerator_config",
        type=str,
        default="dist/accelerator",
        help=(
            "field in the W&B config where the accelerator (i.e. CPU/GPU) is stored "
            "(run.config[accelerator_config]). "
            "if this field is not found int he W&B config GPU is assumed"
        ),
    )
    return vars(parser.parse_args())


def main(host, entity, project, startdate, enddate, world_size_config, default_world_size, accelerator_config):
    print("CLI arguments")
    print(f"host: {host}")
    print(f"entity: {entity}")
    print(f"project: {project}")
    print(f"world_size_config: {world_size_config}")
    print(f"default_world_size: {default_world_size}")
    print(f"accelerator_config: {accelerator_config}")
    # parse dates
    if startdate is not None:
        print(f"startdate: {startdate}")
        startdate = datetime.strptime(startdate, "%Y-%m-%d")
    if enddate is not None:
        print(f"enddate: {enddate}")
        enddate = datetime.strptime(enddate, "%Y-%m-%d")
    print("-" * 50)

    wandb.login(host=host)
    api = wandb.Api()
    data = []
    running_runs = 0
    runs_stuck_at_running = 0
    runs_before_startdate = 0
    runs_after_enddate = 0
    one_day_ago = datetime.now() - timedelta(days=1)
    for run in tqdm(api.runs(f"{entity}/{project}")):
        start = datetime.strptime(run.createdAt, "%Y-%m-%dT%H:%M:%S")
        end = datetime.strptime(run.heartbeatAt, "%Y-%m-%dT%H:%M:%S")
        if startdate is not None and end < startdate:
            runs_before_startdate += 1
            continue
        if enddate is not None and enddate < end:
            runs_after_enddate += 1
            continue
        if run.state == "running":
            # sometimes runs remain running despite being finished already -> count as finished if end was >1 day ago
            if end < one_day_ago:
                runs_stuck_at_running += 1
            else:
                running_runs += 1
                continue
        data.append(
            dict(
                start=start,
                end=end,
                world_size=run.config.get(world_size_config, default_world_size),
                accelerator=run.config.get(accelerator_config, "gpu").lower(),
            )
        )
    wandb.finish()

    # general statistics
    data = pd.DataFrame(data)
    start = data["start"].min()
    end = data["end"].max()
    print("overall statistics:")
    print(f"statistics from {start:%Y-%m-%d} to {end:%Y-%m-%d} ({(end - start).total_seconds() / 86400:.0f} days)")
    print(f"num_runs: {len(data)}")
    if running_runs > 0:
        print(f"number of runs that are still in progress (not counted towards total): {running_runs}")
    if runs_stuck_at_running > 0:
        print(f"number of runs that are stuck on running (counted towards total): {runs_stuck_at_running}")
    if startdate is not None:
        print(f"runs before startdate: {runs_before_startdate}")
    if enddate is not None:
        print(f"runs after enddate: {runs_after_enddate}")

    # statistics for accelerator
    for accelerator in ["gpu", "cpu"]:
        accelerator_data = data[data["accelerator"] == accelerator]
        prefix = accelerator.upper()
        if len(accelerator_data) == 0:
            print("-" * 50)
            print(f"no {prefix}-runs found")
            continue
        total = ((accelerator_data["end"] - accelerator_data["start"]) * accelerator_data["world_size"]).sum()
        total_hours = total.total_seconds() / 3600
        print("-" * 50)
        print(f"{prefix} statistics")
        print(f"num_runs: {len(accelerator_data)}")
        print(f"total: {total.days} days {total.seconds / 3600:.0f} hours")
        print(f"{prefix}-hours: {total_hours:.0f}")
        print(f"{prefix}-hours per run: {total_hours / len(accelerator_data):.2f}")


if __name__ == "__main__":
    main(**parse_args())
