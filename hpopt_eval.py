import comet_ml

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from pytorch_lightning.core.saving import load_hparams_from_yaml

import mutils


def eval_hpopt(project_name):
    comet_api = comet_ml.api.API()
    exps = comet_api.get("transformer", project_name)
    all_dfs, errors = load_dfs(exps)
    # completed_dfs = {k: df for k, df in all_dfs.items() if len(df) > 4}
    dfs = all_dfs
    maes, dt_stamps, dts = get_bet_1maes(dfs)
    plot_results_over_time(maes, dts, dt_stamps)
    max_acc_mae = plot_improovement_over_time(maes, dts)
    print(f"Number of experiment: {len(dfs)}.")
    print(f"Best valid mae: {max_acc_mae[-1]:.4f}.")
    hparams = get_hparams(all_dfs)
    summary = get_hparams_summary(maes, dts, dt_stamps, hparams)
    plot_hparams_over_time(summary)


def load_dfs(exps):
    errors = []
    all_dfs = {}
    for exp in tqdm(exps):
        meta = exp.get_metadata()
        start_time = dt.datetime.fromtimestamp(meta["startTimeMillis"] / 1000.0)
        end_time = dt.datetime.fromtimestamp(meta["endTimeMillis"] / 1000.0)

        try:
            df = mutils.get_exp_csv(exp.id)
        except FileNotFoundError as e:
            errors.append(e)
        df = mutils.get_epoch_info(df)
        df["dt"] = pd.date_range(start_time, end_time, len(df["valid_1mae"]))
        all_dfs[exp.id] = df
    return all_dfs, errors


def get_bet_1maes(dfs):
    maes = []
    dt_stamps = []
    dts = []
    for k, df in dfs.items():
        i = df["valid_1mae"].argmax()
        maes.append(df["valid_1mae"].iloc[i])
        dt_stamps.append(df["dt"].iloc[i].timestamp())
        dts.append(df["dt"].iloc[i])
    return maes, dt_stamps, dts


def plot_results_over_time(maes, dts, dt_stamps):
    x = dt_stamps
    y = maes
    df = pd.DataFrame(
        {"x": np.unique(dts), "y": np.poly1d(np.polyfit(x, y, 1))(np.unique(x))}
    )
    plt.plot(df["x"], df["y"], color="r")
    df = pd.DataFrame({"dt": dts, "valid_1mae": maes})
    plt.scatter(df["dt"], df["valid_1mae"])
    # plt.ylim(0.88, 0.905)
    plt.show()


def plot_improovement_over_time(maes, dts):
    dt_sorted = sorted(list(zip(dts, maes)))
    dt_sorted_mae = [mae for _, mae in dt_sorted]
    dt_sorted_st = [dt for dt, _ in dt_sorted]
    max_acc_mae = np.maximum.accumulate(dt_sorted_mae)
    df = pd.DataFrame({"x": dt_sorted_st, "y": max_acc_mae})
    plt.plot(df["x"], df["y"], color="r")
    df = pd.DataFrame({"x": dts, "y": maes})
    plt.scatter(df["x"], df["y"])
    plt.ylim(0.88, 0.905)
    plt.show()
    return max_acc_mae


def get_hparams(dfs):
    hparams = []
    for k, df in dfs.items():
        f = Path("logs") / "csv" / k / "version_0" / "hparams.yaml"
        hparams.append(load_hparams_from_yaml(f))
    return hparams


def get_hparams_summary(maes, dts, dt_stamps, hparams):
    summary = []
    for hp, mae, dt, dt_stamp in zip(hparams, maes, dts, dt_stamps):
        combined = hp
        combined["best_valid_1mae"] = mae
        combined["dt"] = dt
        combined["dt_stamp"] = dt_stamp
        summary.append(combined)
    return pd.DataFrame(summary)


def plot_hparams_over_time(summary):
    hparam = "project_dim"
    for hparam in [
        "attn_dropout",
        "attn_dropout_a",
        "attn_dropout_v",
        "embed_dropout",
        "out_dropout",
        "relu_dropout",
        "res_dropout",
        "layers",
        "num_heads",
        "lr",
        "project_dim",
    ]:
        if hparam == "lr":
            plt.scatter(np.log10(summary[hparam]), summary["best_valid_1mae"])
            plt.xlabel("log10 lr")
        else:
            plt.scatter(summary[hparam], summary["best_valid_1mae"])
            plt.xlabel(hparam)
        plt.ylabel("best_valid_1mae")
        plt.ylim(0.88, 0.905)
        plt.show()
    # plt.scatter(summary['dt_stamp'], summary[hparam])
    # plt.xlabel('time')
    # plt.ylabel(hparam)
    # plt.show()
