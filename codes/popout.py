

import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
P1 = 1
P2 = 2
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
WINNER_DRAW = "draw"


class Move:
    def __init__(self, column, kind):
        if kind not in ("drop", "pop"):
            raise ValueError(f"invalid kind: {kind!r}")
        if not 0 <= column < COLS:
            raise ValueError(f"column out of [0,{COLS-1}]: {column}")
        self.column = column
        self.kind = kind

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.column == other.column and self.kind == other.kind

    def __hash__(self):
        return hash((self.column, self.kind))

    def __str__(self):
        return f"{self.kind}({self.column})"

    def __repr__(self):
        return f"Move({self.column}, {self.kind!r})"


class State:
    def __init__(self, board, player_to_move, history_counts=None, last_move=None, winner=None):
        self.board = board
        self.player_to_move = player_to_move
        self.history_counts = history_counts if history_counts is not None else {}
        self.last_move = last_move
        self.winner = winner


def state_key(board, player_to_move):
    return board.tobytes() + bytes([player_to_move])


def initial_state():
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    history = {state_key(board, P1): 1}
    return State(board=board, player_to_move=P1, history_counts=history)


def legal_moves(state):
    if state.winner is not None:
        return []
    moves = []
    for c in range(COLS):
        if state.board[0, c] == EMPTY:
            moves.append(Move(c, "drop"))
        if state.board[ROWS - 1, c] == state.player_to_move:
            moves.append(Move(c, "pop"))
    return moves


def _drop_row(board, c):
    for r in range(ROWS - 1, -1, -1):
        if board[r, c] == EMPTY:
            return r
    return -1


def _four_in_a_row_for(board, player):
    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] != player:
                continue
            for dr, dc in DIRECTIONS:
                rr, cc = r + 3 * dr, c + 3 * dc
                if 0 <= rr < ROWS and 0 <= cc < COLS:
                    if all(board[r + i * dr, c + i * dc] == player for i in range(4)):
                        return True
    return False


def _has_legal_pop(board, player):
    return bool((board[ROWS - 1, :] == player).any())


def _has_legal_drop(board):
    return bool((board[0, :] == EMPTY).any())


def check_win(board, last_move, mover):
    other = 3 - mover
    me_won = _four_in_a_row_for(board, mover)
    other_won = _four_in_a_row_for(board, other)

    if last_move.kind == "pop":
        if me_won:
            return mover
        if other_won:
            return other
    else:
        if me_won:
            return mover

    next_player = 3 - mover
    if not _has_legal_drop(board) and not _has_legal_pop(board, next_player):
        return WINNER_DRAW
    return None


def apply_move(state, move):
    if state.winner is not None:
        raise ValueError("Game already finished.")

    new_board = state.board.copy()
    mover = state.player_to_move

    if move.kind == "drop":
        r = _drop_row(new_board, move.column)
        if r == -1:
            raise ValueError(f"Column {move.column} is full.")
        new_board[r, move.column] = mover
    else:
        # pop
        if new_board[ROWS - 1, move.column] != mover:
            raise ValueError(f"Illegal pop on column {move.column}: bottom not player {mover}.")
        new_board[1:ROWS, move.column] = state.board[0:ROWS - 1, move.column]
        new_board[0, move.column] = EMPTY

    new_player = 3 - mover
    new_history = dict(state.history_counts)
    key = state_key(new_board, new_player)
    new_history[key] = new_history.get(key, 0) + 1

    winner = check_win(new_board, move, mover)

    return State(
        board=new_board,
        player_to_move=new_player,
        history_counts=new_history,
        last_move=move,
        winner=winner,
    )


def can_claim_repetition_draw(state):
    key = state_key(state.board, state.player_to_move)
    return state.history_counts.get(key, 0) >= 3


def render(board):
    glyph = {EMPTY: "-", P1: "X", P2: "O"}
    lines = ["".join(glyph[int(v)] for v in row) for row in board]
    return "\n".join(lines)
