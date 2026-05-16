

import argparse
import csv
import math
import os
import random
import sys
import time

from popout import COLS, Move, apply_move, initial_state, legal_moves
from mcts import mcts_strategy


def encode_move(move):
    return f"{'d' if move.kind == 'drop' else 'p'}{move.column}"


def encode_state_row(state):
    flat = state.board.reshape(-1).tolist()
    return [int(v) for v in flat] + [int(state.player_to_move)]


def feature_columns():
    return [f"s{i}" for i in range(42)] + ["to_play"]


def write_header(writer):
    writer.writerow(feature_columns() + ["move"])


def generate_dataset(n_games=50, out_path="popout_dataset.csv", epsilon=0.10,
                     n_simulations=200, rollout="heuristic_win", tactical_root=True,
                     c=math.sqrt(2), seed=0, max_turns=300, verbose=True):
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rng_master = random.Random(seed)
    n_pairs = 0
    n_random_moves = 0
    n_mcts_moves = 0
    n_pops = 0
    game_lengths = []
    winners = {1: 0, 2: 0, "draw": 0, "incomplete": 0}
    move_class_counts = {}
    t0 = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        write_header(writer)

        for g in range(n_games):
            state = initial_state()
            game_seed = rng_master.randint(0, 10**9)
            rng_game = random.Random(game_seed)
            mcts = mcts_strategy(
                n_simulations=n_simulations, c=c,
                rollout=rollout, tactical_root=tactical_root,
                rng=random.Random(game_seed + 1),
            )

            turns = 0
            while state.winner is None and turns < max_turns:
                legal = legal_moves(state)
                if not legal:
                    break
                if rng_game.random() < epsilon:
                    move = legal[rng_game.randrange(len(legal))]
                    n_random_moves += 1
                else:
                    move = mcts(state)
                    n_mcts_moves += 1

                writer.writerow(encode_state_row(state) + [encode_move(move)])
                n_pairs += 1
                cls = encode_move(move)
                move_class_counts[cls] = move_class_counts.get(cls, 0) + 1
                if move.kind == "pop":
                    n_pops += 1

                state = apply_move(state, move)
                turns += 1

            game_lengths.append(turns)
            if state.winner in (1, 2):
                winners[state.winner] += 1
            elif state.winner == "draw":
                winners["draw"] += 1
            else:
                winners["incomplete"] += 1

            if verbose and (g + 1) % max(1, n_games // 10) == 0:
                elapsed = time.time() - t0
                rate = (g + 1) / elapsed
                eta = (n_games - g - 1) / rate if rate > 0 else 0
                print(f"  game {g+1}/{n_games}  pairs={n_pairs}  "
                      f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    csv_size = os.path.getsize(out_path)
    return {
        "n_games": n_games, "n_pairs": n_pairs,
        "n_mcts_moves": n_mcts_moves, "n_random_moves": n_random_moves,
        "n_pops": n_pops, "pop_rate": n_pops / max(1, n_pairs),
        "epsilon": epsilon, "winners": winners,
        "avg_game_length": sum(game_lengths) / max(1, len(game_lengths)),
        "min_game_length": min(game_lengths) if game_lengths else 0,
        "max_game_length": max(game_lengths) if game_lengths else 0,
        "elapsed_s": elapsed, "csv_path": out_path, "csv_bytes": csv_size,
        "move_class_counts": dict(
            sorted(move_class_counts.items(), key=lambda kv: -kv[1])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "popout_dataset.csv"))
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--n-simulations", type=int, default=200)
    parser.add_argument("--rollout", default="heuristic_win",
                        choices=["random", "heuristic_win", "heuristic_block"])
    parser.add_argument("--no-tactical", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=== Generate PopOut dataset ===")
    print(f"games={args.games} eps={args.epsilon} N={args.n_simulations} "
          f"rollout={args.rollout} tactical={not args.no_tactical} seed={args.seed}\n")
    stats = generate_dataset(
        n_games=args.games, out_path=args.out, epsilon=args.epsilon,
        n_simulations=args.n_simulations, rollout=args.rollout,
        tactical_root=not args.no_tactical, seed=args.seed,
    )
    print(f"\nPairs:        {stats['n_pairs']}  (mcts={stats['n_mcts_moves']} "
          f"random={stats['n_random_moves']})")
    print(f"Pops:         {stats['n_pops']} ({stats['pop_rate']*100:.1f}%)")
    print(f"Winners:      P1={stats['winners'][1]} P2={stats['winners'][2]} "
          f"draw={stats['winners']['draw']} incomplete={stats['winners']['incomplete']}")
    print(f"Game length:  avg {stats['avg_game_length']:.1f} plies "
          f"(min={stats['min_game_length']} max={stats['max_game_length']})")
    print(f"Time:         {stats['elapsed_s']:.1f}s")
    print(f"CSV:          {stats['csv_path']}  ({stats['csv_bytes']/1024:.1f} KB)")
    print("\nTop classes:")
    for cls, n in list(stats["move_class_counts"].items())[:8]:
        print(f"  {cls}: {n} ({n/stats['n_pairs']*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
