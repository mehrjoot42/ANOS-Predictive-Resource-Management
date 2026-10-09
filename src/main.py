"""
ANOS-NetBoost: Main entry point.

Orchestrates the full experiment pipeline:
    1. Collect real system telemetry.
    2. Train three model families (LSTM, MLP, Transformer).
    3. Evaluate prediction accuracy against a moving-average baseline.
    4. Execute intelligent priority reduction.
    5. Restore all modified priorities upon exit (Graceful Lifecycle).

Usage:
    python src/main.py
"""
import os
import sys
import time
import csv

# Ensure the src/ directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from monitor import NetworkMonitor
from predictor import NetLSTM, NetMLP, NetTransformer
from optimizer import NetworkOptimizer
from utils import set_seed, compute_metrics, baseline_ma, create_sequences


# ============================================================
# Configuration
# ============================================================
SEEDS = [42, 123, 2026, 7, 99]
SEQ_LEN = 15
EPOCHS = 400
LR = 5e-4
HIDDEN_LSTM = 64
HIDDEN_MLP = 128
HIDDEN_TR = 64
COLLECT_SECONDS = 150
SAMPLE_INTERVAL = 0.5


# ============================================================
# Training helper
# ============================================================
def train_model(model, X_n, Y_n, epochs=EPOCHS, lr=LR):
    """Train a model on normalized sequences."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_n, dtype=torch.float32)
    Y_t = torch.tensor(Y_n, dtype=torch.float32).reshape(-1, 1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, Y_t)
        loss.backward()
        optimizer.step()
    return model


def evaluate_model(model, X_n, Y_std, Y_mean):
    """Return raw (denormalized) predictions."""
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.tensor(X_n, dtype=torch.float32)).numpy().flatten()
    return np.clip(pred_n * Y_std + Y_mean, 0, None)


# ============================================================
# Single run for one seed
# ============================================================
def single_run(seed, samples, verbose=True):
    """Train all three models on the same data and return metrics."""
    set_seed(seed)

    # Build the multivariate time series
    dl = np.array([s["dl"] for s in samples], dtype=np.float32)
    ul = np.array([s["ul"] for s in samples], dtype=np.float32)
    jit = np.array([s["jitter"] for s in samples], dtype=np.float32)
    multi = np.stack([dl, ul, jit], axis=1)

    X, Y = create_sequences(multi, SEQ_LEN)

    # Normalize
    X_mean = X.reshape(-1, X.shape[-1]).mean(0)
    X_std = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    Y_mean = float(Y.mean())
    Y_std = float(Y.std() + 1e-6)
    X_n = (X - X_mean) / X_std
    Y_n = (Y - Y_mean) / Y_std

    results = {}

    # LSTM
    if verbose:
        print(f"  [Seed={seed}] Training LSTM...")
    lstm = NetLSTM(input_dim=3, hidden_dim=HIDDEN_LSTM, num_layers=2)
    train_model(lstm, X_n, Y_n)
    pred_lstm = evaluate_model(lstm, X_n, Y_std, Y_mean)

    # MLP
    if verbose:
        print(f"  [Seed={seed}] Training MLP...")
    mlp = NetMLP(input_dim=3, seq_len=SEQ_LEN, hidden_dim=HIDDEN_MLP)
    train_model(mlp, X_n, Y_n)
    pred_mlp = evaluate_model(mlp, X_n, Y_std, Y_mean)

    # Transformer
    if verbose:
        print(f"  [Seed={seed}] Training Transformer...")
    tr = NetTransformer(input_dim=3, hidden_dim=HIDDEN_TR, nhead=4, num_layers=2)
    train_model(tr, X_n, Y_n)
    pred_tr = evaluate_model(tr, X_n, Y_std, Y_mean)

    # Baseline
    baseline = baseline_ma(dl, w=SEQ_LEN)

    # Align lengths
    n = min(len(pred_lstm), len(pred_mlp), len(pred_tr), len(baseline), len(Y))
    Y_true = Y[:n]

    results["LSTM"] = compute_metrics(pred_lstm[:n], Y_true)
    results["MLP"] = compute_metrics(pred_mlp[:n], Y_true)
    results["Transformer"] = compute_metrics(pred_tr[:n], Y_true)
    results["Baseline"] = compute_metrics(baseline[:n], Y_true)

    if verbose:
        print(f"\n  [Results for Seed={seed}]")
        print(f"  {'Method':<15s} | {'MAE':<10s} | {'RMSE':<10s}")
        print(f"  {'-' * 40}")
        for m, (mae, rmse) in results.items():
            print(f"  {m:<15s} | {mae:<10.5f} | {rmse:<10.5f}")
        print()

    return results


# ============================================================
# Experiment A: 5-seed robustness
# ============================================================
def robustness_experiment(samples):
    print("\n" + "=" * 66)
    print("  Experiment A: 5-Seed Robustness Study")
    print("=" * 66 + "\n")

    all_results = {
        m: {"mae": [], "rmse": []}
        for m in ["LSTM", "MLP", "Transformer", "Baseline"]
    }

    for i, seed in enumerate(SEEDS, 1):
        print(f"[Run {i}/{len(SEEDS)}] Seed = {seed}")
        res = single_run(seed, samples)
        for m, (mae, rmse) in res.items():
            all_results[m]["mae"].append(mae)
            all_results[m]["rmse"].append(rmse)

    print("=" * 66)
    print("  Final Results: Mean +/- Std over 5 seeds")
    print("=" * 66 + "\n")
    print(f"  {'Method':<15s} | {'MAE (mean +/- std)':<25s} | {'RMSE (mean +/- std)':<25s}")
    print(f"  {'-' * 72}")

    summary = {}
    for m in ["LSTM", "MLP", "Transformer", "Baseline"]:
        maes = np.array(all_results[m]["mae"])
        rmses = np.array(all_results[m]["rmse"])
        summary[m] = {
            "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
            "rmse_mean": float(rmses.mean()), "rmse_std": float(rmses.std()),
        }
        print(f"  {m:<15s} | {maes.mean():.5f} +/- {maes.std():.5f}     "
              f"| {rmses.mean():.5f} +/- {rmses.std():.5f}")

    return summary


# ============================================================
# Experiment C: Intelligent action
# ============================================================
def action_experiment(monitor, optimizer, rounds=3):
    print("\n" + "=" * 66)
    print(f"  Experiment C: Intelligent Action ({rounds} rounds)")
    print("=" * 66)
    print("  [!] Please stress the network during this experiment.\n")

    improvements = []

    for r in range(1, rounds + 1):
        print(f"[Round {r}/{rounds}]")

        # Before
        print("  Measuring speed before (8s)...")
        before = []
        t0 = time.time()
        while time.time() - t0 < 8:
            before.append(monitor.snapshot()["dl"])
            time.sleep(0.5)
        avg_before = float(np.mean(before))
        print(f"    Before: {avg_before:.4f} MB/s")

        # Identify and act
        print("  Identifying high-traffic processes...")
        heavy = optimizer.identify(5)
        for p in heavy:
            print(f"    - {p['name'][:30]:30s} PID={p['pid']:5d} "
                  f"estab={p['estab']:2d} score={p['score']:.1f}")

        print("  Lowering priorities...")
        applied = optimizer.optimize(heavy)
        for a in applied:
            print(f"    [OK] {a['name'][:30]:30s} {a['old']} -> {a['new']}")

        # After
        print("  Measuring speed after (8s)...")
        after = []
        t0 = time.time()
        while time.time() - t0 < 8:
            after.append(monitor.snapshot()["dl"])
            time.sleep(0.5)
        avg_after = float(np.mean(after))
        print(f"    After: {avg_after:.4f} MB/s")

        change = 100 * (avg_after - avg_before) / avg_before if avg_before > 0 else 0.0
        improvements.append(change)
        print(f"  --> Change: {change:+.2f}%\n")

        optimizer.restore()
        time.sleep(2)

    print("=" * 66)
    print(f"  Mean improvement: {np.mean(improvements):+.2f}% "
          f"+/- {np.std(improvements):.2f}%")
    print("=" * 66 + "\n")

    return improvements


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("   ANOS-NetBoost: Predictive Resource Management")
    print("=" * 70 + "\n")

    set_seed(42)
    monitor = NetworkMonitor()
    optimizer = NetworkOptimizer()

    # Phase 1: Initial state
    print("[Phase 1] Initial system state")
    print("-" * 70)
    time.sleep(1)
    s = monitor.snapshot()
    print(f"  CPU: {s['cpu']:.1f}% | RAM: {s['ram']:.1f}% | DL: {s['dl']:.3f} MB/s")

    # Phase 2: Collect telemetry
    print(f"\n[Phase 2] Collecting telemetry ({COLLECT_SECONDS}s)")
    print("-" * 70)
    print("  [i] Please apply network pressure: play HD video, download files...\n")

    samples = []
    t0 = time.time()
    last_log = t0
    while time.time() - t0 < COLLECT_SECONDS:
        samples.append(monitor.snapshot())
        now = time.time()
        if now - last_log >= 20:
            dls = [x["dl"] for x in samples]
            print(f"  [{int(now - t0):3d}s] samples={len(samples):4d} | "
                  f"DL avg={np.mean(dls):.3f} | max={max(dls):.3f} MB/s")
            last_log = now
        time.sleep(SAMPLE_INTERVAL)

    print(f"\n  [OK] Collected {len(samples)} samples.\n")

    # Phase 3: Robustness experiment
    summary = robustness_experiment(samples)

    # Phase 4: Intelligent action
    improvements = action_experiment(monitor, optimizer, rounds=3)

    # Phase 5: Save results
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/robustness_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std"])
        for m, v in summary.items():
            w.writerow([m, f"{v['mae_mean']:.6f}", f"{v['mae_std']:.6f}",
                        f"{v['rmse_mean']:.6f}", f"{v['rmse_std']:.6f}"])

    # Final report
    print("=" * 70)
    print("  Final Report")
    print("=" * 70)
    print("\n  Prediction accuracy (5-seed mean):")
    for m in ["LSTM", "MLP", "Transformer", "Baseline"]:
        v = summary[m]
        print(f"    {m:<15s}: MAE = {v['mae_mean']:.5f} +/- {v['mae_std']:.5f}")

    print(f"\n  Action improvement: "
          f"{np.mean(improvements):+.2f}% +/- {np.std(improvements):.2f}%")
    print(f"\n  Results saved to: data/processed/robustness_results.csv")
    print("=" * 70)
    print("\n  [Done] ANOS-NetBoost completed successfully.")


if __name__ == "__main__":
    main()
