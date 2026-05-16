

import argparse
import math
import random
import time

from game import play_game, random_strategy
from popout import P1, P2
from mcts import mcts_strategy

DEFAULT_C = math.sqrt(2)


class MatchResult:
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


def time_strategy(strat):
    times = []
    def wrapped(state):
        t0 = time.perf_counter()
        m = strat(state)
        times.append(time.perf_counter() - t0)
        return m
    wrapped._times = times
    return wrapped


def run_match(factory_a, factory_b, label_a, label_b,
              n_games=10, max_turns=300, seed_base=0):
    wins_a = 0
    wins_b = 0
    draws = 0
    times_a, times_b = [], []
    for g in range(n_games):
        if g % 2 == 0:
            sa = time_strategy(factory_a())
            sb = time_strategy(factory_b())
            p1, p2, a_player = sa, sb, P1
        else:
            sb = time_strategy(factory_b())
            sa = time_strategy(factory_a())
            p1, p2, a_player = sb, sa, P2
        final = play_game(p1, p2, on_render=lambda _: None,
                          show_intermediate=False, max_turns=max_turns)
        times_a.extend(sa._times)
        times_b.extend(sb._times)
        if final.winner == a_player:
            wins_a += 1
        elif final.winner == "draw":
            draws += 1
        else:
            wins_b += 1
    return MatchResult(
        label_a=label_a, label_b=label_b,
        wins_a=wins_a, wins_b=wins_b, draws=draws,
        avg_time_a=sum(times_a)/max(1, len(times_a)),
        avg_time_b=sum(times_b)/max(1, len(times_b)),
        n_games=n_games,
    )


def factory_mcts(n, c=DEFAULT_C, rollout="random", max_children=None, seed=None):
    def f():
        rng = random.Random(seed)
        return mcts_strategy(n_simulations=n, c=c, rollout=rollout,
                             max_children=max_children, rng=rng)
    return f


def factory_random(seed=None):
    def f():
        return random_strategy(random.Random(seed))
    return f


def experiment_A(quick):
    n_games = 6 if quick else 10
    N = 200
    print(f"\n=== A: rollout policy (N={N}, {n_games} games) ===\n")
    matchups = [
        ("random", "heuristic_win"),
        ("random", "heuristic_block"),
        ("heuristic_win", "heuristic_block"),
    ]
    results = []
    for ra, rb in matchups:
        print(f"  {ra} vs {rb} ...", flush=True)
        r = run_match(
            factory_mcts(N, rollout=ra, seed=0),
            factory_mcts(N, rollout=rb, seed=1),
            label_a=ra, label_b=rb, n_games=n_games,
        )
        print(f"    A {r.wins_a} - {r.wins_b} B (draws {r.draws}) "
              f"| t/move A={r.avg_time_a*1000:.0f}ms B={r.avg_time_b*1000:.0f}ms")
        results.append(r)
    return results


def experiment_B(quick):
    n_games = 4 if quick else 8
    print(f"\n=== B: N (rollout=heuristic_win, {n_games} games) ===\n")
    matchups = [(100, 300), (300, 600), (300, "random_baseline")]
    results = []
    for a, b in matchups:
        if b == "random_baseline":
            la, lb = f"MCTS_N{a}_heur", "random"
            print(f"  {la} vs {lb} ...", flush=True)
            r = run_match(
                factory_mcts(a, rollout="heuristic_win", seed=0),
                factory_random(seed=999),
                label_a=la, label_b=lb, n_games=n_games,
            )
        else:
            la, lb = f"MCTS_N{a}", f"MCTS_N{b}"
            print(f"  {la} vs {lb} ...", flush=True)
            r = run_match(
                factory_mcts(a, rollout="heuristic_win", seed=0),
                factory_mcts(b, rollout="heuristic_win", seed=1),
                label_a=la, label_b=lb, n_games=n_games,
            )
        print(f"    A {r.wins_a} - {r.wins_b} B (draws {r.draws}) "
              f"| t/move A={r.avg_time_a*1000:.0f}ms B={r.avg_time_b*1000:.0f}ms")
        results.append(r)
    return results


def experiment_C(quick):
    n_games = 6 if quick else 12
    N = 200
    print(f"\n=== C: C (N={N}, vs random, {n_games} games) ===\n")
    cs = [0.5, 1.0, math.sqrt(2), 2.0]
    results = []
    for c in cs:
        la = f"MCTS_C{c:.2f}"
        print(f"  {la} vs random ...", flush=True)
        r = run_match(
            factory_mcts(N, c=c, rollout="heuristic_win", seed=0),
            factory_random(seed=999),
            label_a=la, label_b="random", n_games=n_games,
        )
        print(f"    {la} {r.wins_a} - {r.wins_b} random (draws {r.draws}) "
              f"| t/move {r.avg_time_a*1000:.0f}ms")
        results.append(r)
    return results


def experiment_D(quick):
    n_games = 4 if quick else 8
    N = 200
    print(f"\n=== D: max_children (N={N}, vs random, {n_games} games) ===\n")
    ks = [None, 5, 3]
    results = []
    for k in ks:
        la = f"MCTS_k{k}"
        print(f"  {la} vs random ...", flush=True)
        r = run_match(
            factory_mcts(N, rollout="heuristic_win", max_children=k, seed=0),
            factory_random(seed=999),
            label_a=la, label_b="random", n_games=n_games,
        )
        print(f"    {la} {r.wins_a} - {r.wins_b} random (draws {r.draws}) "
              f"| t/move {r.avg_time_a*1000:.0f}ms")
        results.append(r)
    return results


def print_table(title, results):
    print(f"\n## {title}\n")
    print(f"| {'A':<24} | {'B':<22} | A | B | D | t/A (ms) | t/B (ms) |")
    print(f"|{'-'*26}|{'-'*24}|---|---|---|----------|----------|")
    for r in results:
        print(f"| {r.label_a:<24} | {r.label_b:<22} | {r.wins_a} | {r.wins_b} | "
              f"{r.draws} | {r.avg_time_a*1000:>8.1f} | {r.avg_time_b*1000:>8.1f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--exp", choices=["A", "B", "C", "D"], default=None)
    args = parser.parse_args()

    t0 = time.time()
    all_results = {}
    if args.exp in (None, "A"):
        all_results["A"] = experiment_A(args.quick)
    if args.exp in (None, "B"):
        all_results["B"] = experiment_B(args.quick)
    if args.exp in (None, "C"):
        all_results["C"] = experiment_C(args.quick)
    if args.exp in (None, "D"):
        all_results["D"] = experiment_D(args.quick)
    elapsed = time.time() - t0

    print(f"\n\n{'='*70}\nSUMMARY ({elapsed:.0f}s total)\n{'='*70}")
    if "A" in all_results:
        print_table("Experiment A -- rollout policy", all_results["A"])
    if "B" in all_results:
        print_table("Experiment B -- N", all_results["B"])
    if "C" in all_results:
        print_table("Experiment C -- C", all_results["C"])
    if "D" in all_results:
        print_table("Experiment D -- max_children", all_results["D"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
