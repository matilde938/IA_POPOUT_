
import argparse
import math
import os
import pickle
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from game import play_game, random_strategy
from decision_tree_builder import (
    accuracy, id3, predict_batch, tree_size, tree_strategy,
)
from popout import P1, P2
from mcts import mcts_strategy

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CODES_DIR, "popout_dataset.csv")
TREE_PATH = os.path.join(CODES_DIR, "decision_tree.pkl")


def factory_random(seed=999):
    return lambda: random_strategy(random.Random(seed))


def factory_mcts(N, rollout="random", tactical=False, seed=0):
    return lambda: mcts_strategy(
        n_simulations=N, rollout=rollout, tactical_root=tactical,
        rng=random.Random(seed),
    )


def factory_tree(tree):
    return lambda: tree_strategy(tree)


def time_strat(strat):
    times = []
    def w(state):
        t0 = time.perf_counter()
        m = strat(state)
        times.append(time.perf_counter() - t0)
        return m
    w._times = times
    return w


class CellResult:
    def __init__(self, label_a, label_b, wins_a, wins_b, draws,
                 avg_time_a, avg_time_b, n_games):
        self.label_a = label_a
        self.label_b = label_b
        self.wins_a = wins_a
        self.wins_b = wins_b
        self.draws = draws
        self.avg_time_a = avg_time_a
        self.avg_time_b = avg_time_b
        self.n_games = n_games


def run_match(fa, fb, label_a, label_b, n_games=4, max_turns=200):
    wa = 0
    wb = 0
    d = 0
    ta_all, tb_all = [], []
    for g in range(n_games):
        if g % 2 == 0:
            sa = time_strat(fa())
            sb = time_strat(fb())
            p1, p2, ap = sa, sb, P1
        else:
            sb = time_strat(fb())
            sa = time_strat(fa())
            p1, p2, ap = sb, sa, P2
        f = play_game(p1, p2, on_render=lambda _: None,
                      show_intermediate=False, max_turns=max_turns)
        ta_all.extend(sa._times)
        tb_all.extend(sb._times)
        if f.winner == ap:
            wa += 1
        elif f.winner == "draw":
            d += 1
        else:
            wb += 1
    return CellResult(
        label_a=label_a, label_b=label_b,
        wins_a=wa, wins_b=wb, draws=d,
        avg_time_a=sum(ta_all)/max(1, len(ta_all)),
        avg_time_b=sum(tb_all)/max(1, len(tb_all)),
        n_games=n_games,
    )


def win_rate_matrix(agents, n_games=4, max_turns=200):
    labels = list(agents.keys())
    n = len(labels)
    matrix = np.zeros((n, n))
    times = {l: [] for l in labels}

    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            fa = agents[la]
            fb = agents[lb]
            print(f"  {la:>10} vs {lb:<10}", end=" ", flush=True)
            r = run_match(fa, fb, la, lb, n_games=n_games, max_turns=max_turns)
            matrix[i, j] = (r.wins_a + 0.5 * r.draws) / r.n_games
            print(f"-> {r.wins_a}-{r.wins_b}-{r.draws}  "
                  f"(t/move A={r.avg_time_a*1e3:.1f}ms B={r.avg_time_b*1e3:.1f}ms)",
                  flush=True)
            times[la].append(r.avg_time_a)
            times[lb].append(r.avg_time_b)

    avg_times = {l: float(np.mean(times[l])) if times[l] else 0.0 for l in labels}
    return labels, matrix, avg_times


def learning_curve(dataset_path, train_fracs=(0.2, 0.4, 0.6, 0.8),
                   max_depth=10, seed=0):
    df = pd.read_csv(dataset_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    feats = [c for c in df.columns if c != "move"]
    n = len(df)
    test_size = int(0.2 * n)
    test = df.iloc[-test_size:]
    pool = df.iloc[:-test_size]
    rows = []
    for frac in train_fracs:
        size = max(1, int(frac * len(pool)))
        train = pool.iloc[:size]
        tree = id3(train[feats], train["move"], feats, max_depth=max_depth)
        a = accuracy(test["move"], predict_batch(tree, test[feats]))
        sz = tree_size(tree)
        rows.append({"frac": frac, "n_train": size, "test_acc": a,
                     "leaves": sz["leaves"], "depth": sz["depth"]})
        print(f"  frac={frac:.2f} n_train={size:>4} acc_te={a:.3f} "
              f"leaves={sz['leaves']:>3}", flush=True)
    return rows


def chart_winrate_heatmap(labels, matrix, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Opponent (B)")
    ax.set_ylabel("Agent (A)")
    ax.set_title("Win-rate (A vs B) -- 0=B always, 1=A always")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "black" if 0.3 <= matrix[i, j] <= 0.7 else "white"
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color=color, fontsize=10)
    fig.colorbar(im, ax=ax, label="Win rate of A")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_learning_curve(rows, out_path):
    xs = [r["n_train"] for r in rows]
    ys = [r["test_acc"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_xlabel("# training examples")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Tree learning curve (max_depth=10) over PopOut dataset")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    for r in rows:
        ax.text(r["n_train"], r["test_acc"]+0.02, f"{r['test_acc']:.2f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_depth_sensitivity(out_path):
    depths = [3, 5, 8, 10, "None"]
    train_acc = [0.303, 0.526, 0.876, 0.904, 0.904]
    test_acc = [0.169, 0.157, 0.221, 0.227, 0.227]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(depths)))
    ax.plot(x, train_acc, marker="o", label="Train", linewidth=2)
    ax.plot(x, test_acc, marker="s", label="Test", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Depth sensitivity -- Tree over PopOut dataset")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="2 games per cell instead of 4")
    parser.add_argument("--out-dir", default=os.path.join(CODES_DIR, "content"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(TREE_PATH):
        print(f"ERROR: tree not found at {TREE_PATH}")
        print("Run: python train_tree.py --sweep")
        return 1

    with open(TREE_PATH, "rb") as f:
        tree = pickle.load(f)

    n_games = 2 if args.quick else 4

    agents = {
        "Random": factory_random(seed=999),
        "MCTS-E": factory_mcts(N=100, rollout="random", tactical=False, seed=11),
        "MCTS-M": factory_mcts(N=200, rollout="heuristic_win", tactical=True, seed=22),
        "Tree":   factory_tree(tree),
    }

    print(f"=== Phase 8 evaluation ({n_games} games/cell) ===\n--- Win-rate matrix ---")
    t0 = time.time()
    labels, matrix, avg_times = win_rate_matrix(agents, n_games=n_games, max_turns=200)
    print(f"\nMatrix computed in {time.time()-t0:.0f}s.\n")

    print("--- Win-rate table (row beat column) ---")
    print(f"{'':10s}" + "".join(f"{l:>10}" for l in labels))
    for i, l in enumerate(labels):
        row = "".join(f"{matrix[i,j]:>10.2f}" for j in range(len(labels)))
        print(f"{l:10s}{row}")

    print("\n--- Average time per move ---")
    for l, t in avg_times.items():
        if t < 1e-3:
            print(f"  {l}: {t*1e6:.0f} us")
        else:
            print(f"  {l}: {t*1e3:.1f} ms")

    print("\n--- Learning curve ---")
    lc_rows = learning_curve(DATASET_PATH)

    print("\n--- Charts ---")
    chart_winrate_heatmap(labels, matrix,
                          os.path.join(args.out_dir, "winrate_matrix.png"))
    chart_learning_curve(lc_rows,
                         os.path.join(args.out_dir, "tree_learning_curve.png"))
    chart_depth_sensitivity(os.path.join(args.out_dir, "tree_depth_sensitivity.png"))
    for name in ("winrate_matrix.png", "tree_learning_curve.png",
                 "tree_depth_sensitivity.png"):
        print(f"  [saved] {name}")
    print(f"\nAll PNGs in: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
