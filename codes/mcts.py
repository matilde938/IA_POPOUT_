

import math
import random

from popout import COLS, Move, State, apply_move, legal_moves

DEFAULT_C = math.sqrt(2)
DEFAULT_N_SIMULATIONS = 500
ROLLOUT_MAX_DEPTH = 200

CENTER = COLS // 2
COLUMN_PRIORITY = sorted(range(COLS), key=lambda c: abs(c - CENTER))


def _ordered_legal_moves(state, max_children):
    moves = legal_moves(state)
    if max_children is None or len(moves) <= max_children:
        return moves
    drops = [m for m in moves if m.kind == "drop"]
    pops = [m for m in moves if m.kind == "pop"]
    drops.sort(key=lambda m: abs(m.column - CENTER))
    pops.sort(key=lambda m: abs(m.column - CENTER))
    return (drops + pops)[:max_children]


class Node:
    def __init__(self, state, parent=None, move_in=None, max_children=None):
        self.state = state
        self.parent = parent
        self.move_in = move_in
        self.children = {}
        self.untried_moves = list(_ordered_legal_moves(state, max_children))
        self.N = 0
        self.U = 0.0

    def is_terminal(self):
        return self.state.winner is not None

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0 and len(self.children) > 0

    def best_child(self, c):
        log_N_parent = math.log(self.N) if self.N > 0 else 0.0

        def ucb1(child):
            if child.N == 0:
                return math.inf
            exploit = child.U / child.N
            explore = c * math.sqrt(log_N_parent / child.N)
            return exploit + explore

        return max(self.children.values(), key=ucb1)

    def expand(self, rng, max_children=None):
        idx = rng.randrange(len(self.untried_moves))
        move = self.untried_moves.pop(idx)
        next_state = apply_move(self.state, move)
        child = Node(next_state, parent=self, move_in=move, max_children=max_children)
        self.children[move] = child
        return child

    def most_visited_child(self):
        return max(self.children.values(), key=lambda c: c.N)


def _find_winning_move(state, moves):
    me = state.player_to_move
    for m in moves:
        ns = apply_move(state, m)
        if ns.winner == me:
            return m
    return None


def _move_is_safe(state, m):
    ns = apply_move(state, m)
    if ns.winner is not None:
        return True
    opp = ns.player_to_move
    for om in legal_moves(ns):
        nns = apply_move(ns, om)
        if nns.winner == opp:
            return False
    return True


def find_forced_win(state, depth=2):
    if depth < 1 or state.winner is not None:
        return None

    me = state.player_to_move
    moves = legal_moves(state)

    win_now = _find_winning_move(state, moves)
    if win_now is not None:
        return win_now
    if depth == 1:
        return None

    for m in moves:
        ns = apply_move(state, m)
        if ns.winner == me:
            return m
        if ns.winner is not None:
            continue
        opp_moves = legal_moves(ns)
        if not opp_moves:
            continue
        all_lead_to_my_win = True
        for om in opp_moves:
            nns = apply_move(ns, om)
            if nns.winner == me:
                continue
            if nns.winner is not None:
                all_lead_to_my_win = False
                break
            if _find_winning_move(nns, legal_moves(nns)) is None:
                all_lead_to_my_win = False
                break
        if all_lead_to_my_win:
            return m
    return None


def random_playout(state, rng, max_depth=ROLLOUT_MAX_DEPTH):
    depth = 0
    while state.winner is None and depth < max_depth:
        moves = legal_moves(state)
        if not moves:
            return "draw"
        idx = rng.randrange(len(moves))
        state = apply_move(state, moves[idx])
        depth += 1
    return state.winner if state.winner is not None else "draw"


def heuristic_win_playout(state, rng, max_depth=ROLLOUT_MAX_DEPTH):
    depth = 0
    while state.winner is None and depth < max_depth:
        moves = legal_moves(state)
        if not moves:
            return "draw"
        winning = _find_winning_move(state, moves)
        chosen = winning if winning is not None else moves[rng.randrange(len(moves))]
        state = apply_move(state, chosen)
        depth += 1
    return state.winner if state.winner is not None else "draw"


def heuristic_block_playout(state, rng, max_depth=ROLLOUT_MAX_DEPTH):
    depth = 0
    while state.winner is None and depth < max_depth:
        moves = legal_moves(state)
        if not moves:
            return "draw"
        winning = _find_winning_move(state, moves)
        if winning is not None:
            state = apply_move(state, winning)
            depth += 1
            continue
        safe = [m for m in moves if _move_is_safe(state, m)]
        pool = safe if safe else moves
        chosen = pool[rng.randrange(len(pool))]
        state = apply_move(state, chosen)
        depth += 1
    return state.winner if state.winner is not None else "draw"


ROLLOUTS = {
    "random": random_playout,
    "heuristic_win": heuristic_win_playout,
    "heuristic_block": heuristic_block_playout,
}


def backprop(leaf, winner):
    node = leaf
    while node is not None:
        node.N += 1
        if node.parent is not None:
            chooser = node.parent.state.player_to_move
            if winner == chooser:
                node.U += 1.0
            elif winner == "draw":
                node.U += 0.5
        node = node.parent


def mcts_search(root_state, n_simulations=DEFAULT_N_SIMULATIONS, c=DEFAULT_C,
                rollout="random", max_children=None, tactical_root=False,
                tactical_depth=2, rng=None):
    if root_state.winner is not None:
        return None
    moves = legal_moves(root_state)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    if tactical_root:
        forced = find_forced_win(root_state, depth=tactical_depth)
        if forced is not None:
            return forced

    if rollout not in ROLLOUTS:
        raise ValueError(f"invalid rollout: {rollout!r}. Choose from {list(ROLLOUTS)}.")
    playout_fn = ROLLOUTS[rollout]

    if rng is None:
        rng = random.Random()
    root = Node(root_state, max_children=max_children)

    for _ in range(n_simulations):
        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(c)
        if not node.is_terminal():
            node = node.expand(rng, max_children=max_children)
        winner = playout_fn(node.state, rng)
        backprop(node, winner)

    return root.most_visited_child().move_in


def mcts_strategy(n_simulations=DEFAULT_N_SIMULATIONS, c=DEFAULT_C, rollout="random",
                  max_children=None, tactical_root=False, tactical_depth=2, rng=None):
    if rng is None:
        rng = random.Random()

    def strat(state):
        move = mcts_search(
            state,
            n_simulations=n_simulations,
            c=c,
            rollout=rollout,
            max_children=max_children,
            tactical_root=tactical_root,
            tactical_depth=tactical_depth,
            rng=rng,
        )
        if move is None:
            raise RuntimeError("mcts_strategy called on terminal state.")
        return move

    return strat
