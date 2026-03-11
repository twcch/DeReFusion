"""
Bayesian Optimization for DyVolFusion hyperparameters using Optuna (TPE).

Usage:
    pip install optuna
    python run_bayesian_opt.py
"""

import argparse
import os
import random
import time
import json

import numpy as np
import optuna
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def build_args(params: dict) -> argparse.Namespace:
    """Build an argparse.Namespace identical to what run.py produces."""
    args = argparse.Namespace(
        # basic
        task_name="long_term_forecast",
        is_training=1,
        model_id="bayesian_opt",
        model="DyVolFusion",

        # data
        data="custom",
        root_path="./dataset/yfinance/",
        data_path="GSPC-2016-2025.csv",
        features="MS",
        target="Close",
        freq="b",
        checkpoints="./checkpoints/",

        # forecasting
        seq_len=96,
        label_len=48,
        pred_len=24,
        seasonal_patterns="Monthly",
        inverse=False,

        # model
        enc_in=5,
        dec_in=5,
        c_out=1,
        e_layers=params["e_layers"],
        d_layers=params["d_layers"],
        factor=params["factor"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        n_heads=params["n_heads"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        embed="timeF",
        activation="gelu",
        use_norm=1,
        moving_avg=25,
        distil=True,
        channel_independence=1,
        decomp_method="moving_avg",
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        patch_len=16,
        individual=False,

        # optimization
        num_workers=10,
        itr=1,
        train_epochs=params.get("train_epochs", 30),
        batch_size=params.get("batch_size", 32),
        patience=5,
        des="BayesOpt",
        loss="MSE",
        lradj="cosine",
        use_amp=False,

        # GPU
        use_gpu=True,
        no_use_gpu=False,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0",
        device_ids=[0],

        # de-stationary
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,

        # dtw
        use_dtw=False,

        # augmentation
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",

        # GCN / TimeFilter
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        alpha=0.1,
        top_p=0.5,
        pos=1,
    )

    # set device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    return args


def objective(trial: optuna.Trial) -> float:
    """Single Optuna trial: sample hyperparams → train → return vali loss."""

    # ── Search space ──
    params = {
        # 模型結構
        "e_layers": trial.suggest_int("e_layers", 1, 4),
        "d_layers": trial.suggest_int("d_layers", 1, 3),
        "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128]),
        "d_ff": trial.suggest_categorical("d_ff", [32, 64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "factor": trial.suggest_int("factor", 1, 5),

        # 訓練參數
        "dropout": trial.suggest_float("dropout", 0.05, 0.5, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "train_epochs": 30,
    }

    # d_model 必須能被 n_heads 整除
    if params["d_model"] % params["n_heads"] != 0:
        raise optuna.TrialPruned()

    args = build_args(params)

    # setting string (與 run.py 一致)
    setting = (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}"
        f"_df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}"
        f"_eb{args.embed}_dt{args.distil}_{args.des}_{trial.number}"
    )

    exp = Exp_Long_Term_Forecast(args)

    # train & get best vali loss
    from utils.tools import EarlyStopping, adjust_learning_rate
    from data_provider.data_factory import data_provider
    import torch.nn as nn

    train_data, train_loader = exp._get_data(flag="train")
    vali_data, vali_loader = exp._get_data(flag="val")

    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    model_optim = exp._select_optimizer()
    criterion = exp._select_criterion()

    best_vali_loss = float("inf")

    for epoch in range(args.train_epochs):
        exp.model.train()
        train_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        vali_loss = exp.vali(vali_data, vali_loader, criterion)
        best_vali_loss = min(best_vali_loss, vali_loss)

        # Report to Optuna for pruning
        trial.report(vali_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        early_stopping(vali_loss, exp.model, path)
        if early_stopping.early_stop:
            break

        adjust_learning_rate(model_optim, epoch + 1, args)

    # cleanup checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Trial {trial.number}: params={params}, best_vali_loss={best_vali_loss:.6f}")
    return best_vali_loss


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # ── Optuna study ──
    study = optuna.create_study(
        study_name="DyVolFusion_hpo",
        direction="minimize",            # 最小化 vali loss
        sampler=optuna.samplers.TPESampler(seed=fix_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage="sqlite:///bayesian_opt.db",  # 可斷點續跑
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=50, gc_after_trial=True)

    # ── Results ──
    print("\n" + "=" * 60)
    print("Best trial:")
    best = study.best_trial
    print(f"  Vali Loss: {best.value:.6f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # 儲存結果
    results = {
        "best_value": best.value,
        "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open("bayesian_opt_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nResults saved to bayesian_opt_results.json")
    print(f"Optuna DB saved to bayesian_opt.db (可用 optuna-dashboard 視覺化)")


if __name__ == "__main__":
    main()
