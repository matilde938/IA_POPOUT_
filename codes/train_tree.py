import argparse
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd

from game import play_game, random_strategy
from decision_tree_builder import (
    accuracy, id3, predict_batch, render_tree_matplotlib, tree_size,
    tree_strategy,
)
from popout import P1, P2
from mcts import mcts_strategy

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CODES_DIR, "popout_dataset.csv")
TREE_PATH = os.path.join(CODES_DIR, "decision_tree.pkl")


def load_dataset(seed=0, frac_train=0.8):
    df = pd.read_csv(DATASET_PATH)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != "move"]
    sp = int(frac_train * len(df))
    train, test = df.iloc[:sp], df.iloc[sp:]
    return (train[feature_cols], train["move"],
            test[feature_cols], test["move"], feature_cols)


def train(max_depth=None, seed=0):
    X_tr, y_tr, X_te, y_te, feats = load_dataset(seed=seed)
    t0 = time.time()
    tree = id3(X_tr, y_tr, feats, max_depth=max_depth)
    train_t = time.time() - t0
    return {
        "tree": tree,
        "train_accuracy": accuracy(y_tr, predict_batch(tree, X_tr)),
        "test_accuracy": accuracy(y_te, predict_batch(tree, X_te)),
        "size": tree_size(tree),
        "train_time_s": train_t,
        "n_train": len(X_tr), "n_test": len(X_te),
        "feature_cols": feats,
    }


def sweep_max_depth(depths=(3, 5, 8, 10, None), seed=0):
    print(f"{'max_depth':>10} | {'train':>6} | {'test':>6} | "
          f"{'leaves':>6} | {'depth':>5}")
    print("-" * 50)
    rows = []
    for d in depths:
        r = train(max_depth=d, seed=seed)
        print(f"{str(d):>10} | {r['train_accuracy']:>6.3f} | "
              f"{r['test_accuracy']:>6.3f} | {r['size']['leaves']:>6} | "
              f"{r['size']['depth']:>5}")
        rows.append((d, r))
    return rows


def play_match(strat_a_factory, strat_b_factory, n_games=10, max_turns=300):
    wins_a = 0
    wins_b = 0
    draws = 0
    times_a, times_b = [], []
    for g in range(n_games):
        def mk(fac, store):
            s = fac()
            def w(state):
                t0 = time.perf_counter()
                m = s(state)
                store.append(time.perf_counter() - t0)
                return m
            return w

        if g % 2 == 0:
            sa = mk(strat_a_factory, times_a)
            sb = mk(strat_b_factory, times_b)
            p1, p2, a_player = sa, sb, P1
        else:
            sb = mk(strat_b_factory, times_b)
            sa = mk(strat_a_factory, times_a)
            p1, p2, a_player = sb, sa, P2

        final = play_game(p1, p2, on_render=lambda _: None,
                          show_intermediate=False, max_turns=max_turns)
        if final.winner == a_player:
            wins_a += 1
        elif final.winner == "draw":
            draws += 1
        else:
            wins_b += 1
    return {
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "avg_time_a": sum(times_a) / max(1, len(times_a)),
        "avg_time_b": sum(times_b) / max(1, len(times_b)),
    }


def _truncate_tree(tree, max_depth):
    from decision_tree_builder import Node

    def copy(node, d):
        if node.is_leaf or d >= max_depth:
            counts = node.class_counts
            label = node.label
            if label is None and counts:
                label = max(counts.items(), key=lambda kv: kv[1])[0]
            return Node(label=label, n_samples=node.n_samples,
                        class_counts=node.class_counts)
        new = Node(feature=node.feature, n_samples=node.n_samples,
                   class_counts=node.class_counts)
        for k, ch in node.children.items():
            new.children[k] = copy(ch, d + 1)
        return new

    return copy(tree, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--vs-random", type=int, default=0)
    parser.add_argument("--vs-mcts", type=int, default=0)
    parser.add_argument("--save", default=TREE_PATH)
    parser.add_argument("--out-dir", default=os.path.join(CODES_DIR, "content"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== ID3 over PopOut dataset ===")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: dataset not found at {DATASET_PATH}")
        return 1

    if args.sweep:
        print("\n--- max_depth sweep ---")
        rows = sweep_max_depth()
        best_d, best_r = max(rows, key=lambda kv: kv[1]["test_accuracy"])
        print(f"\nBest max_depth: {best_d}  (acc_te={best_r['test_accuracy']:.3f})")
        result = best_r
    else:
        print(f"\n--- training with max_depth={args.max_depth} ---")
        result = train(max_depth=args.max_depth)
        print(f"  train_accuracy = {result['train_accuracy']:.3f}")
        print(f"  test_accuracy  = {result['test_accuracy']:.3f}")
        print(f"  leaves={result['size']['leaves']} depth={result['size']['depth']}")
        print(f"  train_time={result['train_time_s']:.2f}s")

    tree = result["tree"]
    with open(args.save, "wb") as f:
        pickle.dump(tree, f)
    print(f"\n[saved] {args.save}")

    try:
        trunc = _truncate_tree(tree, max_depth=3)
        fig = render_tree_matplotlib(trunc, figsize=(16, 8))
        out_png = os.path.join(args.out_dir, "popout_tree_top3.png")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"[saved] {out_png}")
    except Exception as e:
        print(f"[warn] matplotlib failed: {e}")

    if args.vs_random > 0:
        print(f"\n--- Tree vs Random ({args.vs_random} games) ---")
        r = play_match(lambda: tree_strategy(tree),
                       lambda: random_strategy(random.Random(0)),
                       n_games=args.vs_random)
        print(f"  Tree {r['wins_a']} - {r['wins_b']} Random  (d={r['draws']})  "
              f"| t/move tree={r['avg_time_a']*1e6:.0f}us "
              f"random={r['avg_time_b']*1e6:.0f}us")

    if args.vs_mcts > 0:
        print(f"\n--- Tree vs MCTS-Medium ({args.vs_mcts} games) ---")
        r = play_match(
            lambda: tree_strategy(tree),
            lambda: mcts_strategy(n_simulations=200, rollout="heuristic_win",
                                  tactical_root=True, rng=random.Random(99)),
            n_games=args.vs_mcts,
        )
        print(f"  Tree {r['wins_a']} - {r['wins_b']} MCTS  (d={r['draws']})  "
              f"| t/move tree={r['avg_time_a']*1e6:.0f}us "
              f"mcts={r['avg_time_b']*1e3:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
