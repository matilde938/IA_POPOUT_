
import os
import random

from popout import (
    COLS, EMPTY, Move, P1, P2, State,
    apply_move, can_claim_repetition_draw, initial_state, legal_moves, render,
)

TREE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_tree.pkl")

EASY = (100, "random", False)
MEDIUM = (400, "heuristic_win", True)
HARD = (800, "heuristic_win", True)


HELP_TEXT = """\
Commands:
  0..6           drop in column (shortcut)
  d <col>        explicit drop  (e.g. d 3)
  p <col>        pop            (e.g. p 0)
  draw           claim draw by triple repetition (when applicable)
  q | quit       resign
  ? | help       show this help
"""


class ParseError(Exception):
    pass


def format_state(state):
    header = " " + "".join(str(c) for c in range(COLS))
    body = "\n".join(" " + line for line in render(state.board).split("\n"))
    glyph = {P1: "X", P2: "O"}
    if state.winner is None:
        footer = f"P{state.player_to_move} ({glyph[state.player_to_move]}) to move."
    elif state.winner == "draw":
        footer = "Draw."
    else:
        footer = f"P{state.winner} ({glyph[state.winner]}) wins!"
    return f"{header}\n{body}\n{footer}"


def parse_human_input(text, state):
    t = text.strip().lower()
    if t == "":
        raise ParseError("Empty input. Type '?' for help.")
    if t in ("q", "quit"):
        return "resign"
    if t == "draw":
        if can_claim_repetition_draw(state):
            return "draw"
        raise ParseError("Triple repetition not reached -- cannot claim draw.")
    if t in ("?", "help"):
        raise ParseError(HELP_TEXT)
    if len(t) == 1 and t.isdigit():
        return Move(int(t), "drop")
    parts = t.split()
    if len(parts) == 2 and parts[0] in ("d", "drop", "p", "pop"):
        if not parts[1].isdigit():
            raise ParseError(f"Column is not a number: {parts[1]!r}.")
        col = int(parts[1])
        if not 0 <= col < COLS:
            raise ParseError(f"Column out of [0,{COLS-1}]: {col}.")
        kind = "drop" if parts[0] in ("d", "drop") else "pop"
        return Move(col, kind)
    raise ParseError(f"Invalid input: {text!r}. Type '?' for help.")


def human_strategy(state, input_fn=input, output_fn=print):
    while True:
        try:
            raw = input_fn(f"P{state.player_to_move}> ")
        except EOFError:
            return "resign"
        try:
            decision = parse_human_input(raw, state)
        except ParseError as exc:
            output_fn(str(exc))
            continue
        if isinstance(decision, str):
            return decision
        if decision not in legal_moves(state):
            output_fn(f"Illegal move: {decision}. Try another.")
            continue
        return decision


def random_strategy(rng=None):
    if rng is None:
        rng = random.Random()

    def strat(state):
        moves = legal_moves(state)
        if not moves:
            return "resign"
        return rng.choice(moves)

    return strat


def scripted_strategy(decisions):
    it = iter(decisions)

    def strat(state):
        return next(it)

    return strat


def play_game(p1, p2, on_render=print, max_turns=300, show_intermediate=True):
    state = initial_state()
    strategies = {P1: p1, P2: p2}
    if show_intermediate:
        on_render(format_state(state))

    turns = 0
    while state.winner is None:
        if turns >= max_turns:
            return State(
                board=state.board,
                player_to_move=state.player_to_move,
                history_counts=state.history_counts,
                last_move=state.last_move,
                winner="draw",
            )
        mover = state.player_to_move
        decision = strategies[mover](state)

        if decision == "resign":
            other = 3 - mover
            return State(
                board=state.board,
                player_to_move=state.player_to_move,
                history_counts=state.history_counts,
                last_move=state.last_move,
                winner=other,
            )
        if decision == "draw":
            return State(
                board=state.board,
                player_to_move=state.player_to_move,
                history_counts=state.history_counts,
                last_move=state.last_move,
                winner="draw",
            )

        state = apply_move(state, decision)
        turns += 1
        if show_intermediate:
            on_render(format_state(state))

    return state


def _announce(strat, label):
    def w(state):
        print(f"[{label} thinking...]", flush=True)
        return strat(state)
    return w


def _make_mcts(preset, label, seed=None):
    from mcts import mcts_strategy
    n_sims, rollout, tactical = preset
    rng = random.Random(seed)
    strat = mcts_strategy(n_simulations=n_sims, rollout=rollout,
                          tactical_root=tactical, rng=rng)
    return _announce(strat, label)


def _make_tree():
    import pickle
    from decision_tree_builder import tree_strategy
    with open(TREE_PATH, "rb") as f:
        tree = pickle.load(f)
    return tree_strategy(tree)


def _build_modes():
    modes = [
        ("Human vs Human",
         lambda: (human_strategy, human_strategy)),
        ("Human vs MCTS -- Easy",
         lambda: (human_strategy, _make_mcts(EASY, "MCTS-Easy"))),
        ("Human vs MCTS -- Medium",
         lambda: (human_strategy, _make_mcts(MEDIUM, "MCTS-Medium"))),
        ("Human vs MCTS -- Hard",
         lambda: (human_strategy, _make_mcts(HARD, "MCTS-Hard"))),
        ("MCTS vs MCTS",
         lambda: (_make_mcts(MEDIUM, "MCTS-1"),
                  _make_mcts(MEDIUM, "MCTS-2", seed=99))),
    ]
    if os.path.exists(TREE_PATH):
        modes.extend([
            ("Human vs Tree (ID3)",
             lambda: (human_strategy, _make_tree())),
            ("MCTS vs Tree (CvC, 2 algos)",
             lambda: (_make_mcts(MEDIUM, "MCTS"), _make_tree())),
        ])
    return modes


def _print_menu(modes):
    print("\nPopOut CLI -- choose mode:")
    for i, (label, _) in enumerate(modes):
        print(f"  {i+1}) {label}")
    print("  q) quit")


def _read_mode_choice(modes):
    while True:
        try:
            raw = input("Mode> ").strip().lower()
        except EOFError:
            return None
        if raw in ("q", "quit", "exit"):
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(modes):
                return idx
        print(f"Invalid choice. Pick 1-{len(modes)} or 'q'.")


def main():
    modes = _build_modes()
    if not os.path.exists(TREE_PATH):
        print("(decision_tree.pkl not found - Tree modes unavailable. "
              "Run: python train_tree.py --sweep)")
    while True:
        _print_menu(modes)
        idx = _read_mode_choice(modes)
        if idx is None:
            return
        label, factory = modes[idx]
        print(f"\n=== {label} ===")
        print(HELP_TEXT)
        p1, p2 = factory()
        final = play_game(p1, p2)
        print(format_state(final))
        try:
            again = input("\nPlay again? [y/N] ").strip().lower()
        except EOFError:
            return
        if again not in ("y", "yes"):
            return


if __name__ == "__main__":
    main()
