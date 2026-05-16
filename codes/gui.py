
import os
import sys
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from popout import (  # noqa: E402
    COLS, EMPTY, P1, P2, ROWS, Move,
    apply_move, can_claim_repetition_draw, initial_state, legal_moves,
)


CELL = 90
PAD = 12
HEADER = 70
FOOTER = 70
STATUS = 50
WIDTH = COLS * CELL + 2 * PAD
HEIGHT = STATUS + HEADER + ROWS * CELL + FOOTER + PAD

BG = (24, 36, 64)
HOLE = (16, 24, 44)
P1_COLOR = (235, 84, 80)
P2_COLOR = (255, 207, 64)
TEXT = (240, 240, 240)
MUTED = (140, 150, 170)
ACCENT = (90, 180, 255)
WIN_HIGHLIGHT = (60, 220, 120)
DROP_BTN = (60, 80, 130)
POP_BTN_OK = (180, 80, 80)
POP_BTN_DISABLED = (60, 60, 80)
HOVER = (110, 140, 200)


def col_x(c):
    return PAD + c * CELL


def cell_rect(r, c):
    return pygame.Rect(col_x(c), STATUS + HEADER + r * CELL, CELL, CELL)


def drop_btn_rect(c):
    return pygame.Rect(col_x(c) + 6, STATUS + 6, CELL - 12, HEADER - 12)


def pop_btn_rect(c):
    y = STATUS + HEADER + ROWS * CELL + 6
    return pygame.Rect(col_x(c) + 6, y, CELL - 12, FOOTER - 12)


def draw_board(screen, font_small, state, mouse_pos, busy, winning_cells=None):
    screen.fill((10, 14, 28))
    winning_cells = set(winning_cells or [])

    legals = set(legal_moves(state)) if state.winner is None and not busy else set()
    legal_drop_cols = {m.column for m in legals if m.kind == "drop"}
    legal_pop_cols = {m.column for m in legals if m.kind == "pop"}

    for c in range(COLS):
        rect = drop_btn_rect(c)
        is_legal = c in legal_drop_cols
        is_hover = is_legal and rect.collidepoint(mouse_pos)
        color = HOVER if is_hover else (DROP_BTN if is_legal else POP_BTN_DISABLED)
        pygame.draw.rect(screen, color, rect, border_radius=8)
        label = font_small.render("Drop", True, TEXT if is_legal else MUTED)
        screen.blit(label, label.get_rect(center=rect.center))

    board_rect = pygame.Rect(PAD, STATUS + HEADER, COLS * CELL, ROWS * CELL)
    pygame.draw.rect(screen, BG, board_rect, border_radius=12)

    for r in range(ROWS):
        for c in range(COLS):
            cx = col_x(c) + CELL // 2
            cy = STATUS + HEADER + r * CELL + CELL // 2
            radius = CELL // 2 - 8
            v = int(state.board[r, c])
            if v == EMPTY:
                pygame.draw.circle(screen, HOLE, (cx, cy), radius)
            else:
                fill = P1_COLOR if v == P1 else P2_COLOR
                pygame.draw.circle(screen, fill, (cx, cy), radius)
                if (r, c) in winning_cells:
                    pygame.draw.circle(screen, WIN_HIGHLIGHT, (cx, cy), radius, width=4)

    for c in range(COLS):
        rect = pop_btn_rect(c)
        is_legal = c in legal_pop_cols
        is_hover = is_legal and rect.collidepoint(mouse_pos)
        color = HOVER if is_hover else (POP_BTN_OK if is_legal else POP_BTN_DISABLED)
        pygame.draw.rect(screen, color, rect, border_radius=8)
        label = font_small.render("Pop", True, TEXT if is_legal else MUTED)
        screen.blit(label, label.get_rect(center=rect.center))


def find_winning_cells(state):
    if state.winner not in (P1, P2):
        return []
    player = state.winner
    DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(ROWS):
        for c in range(COLS):
            if int(state.board[r, c]) != player:
                continue
            for dr, dc in DIRS:
                rr, cc = r + 3 * dr, c + 3 * dc
                if 0 <= rr < ROWS and 0 <= cc < COLS:
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if all(int(state.board[y, x]) == player for y, x in cells):
                        return cells
    return []


def draw_status(screen, font, state, mode_label, busy):
    bar = pygame.Rect(0, 0, WIDTH, STATUS)
    pygame.draw.rect(screen, (18, 24, 44), bar)
    if state.winner is None:
        glyph_color = P1_COLOR if state.player_to_move == P1 else P2_COLOR
        msg = f"P{state.player_to_move} to move"
        if busy:
            msg += "  -  AI thinking..."
    elif state.winner == "draw":
        glyph_color = MUTED
        msg = "Draw."
    else:
        glyph_color = P1_COLOR if state.winner == P1 else P2_COLOR
        msg = f"P{state.winner} wins!"
    pygame.draw.circle(screen, glyph_color, (24, STATUS // 2), 12)
    text = font.render(msg, True, TEXT)
    screen.blit(text, text.get_rect(midleft=(48, STATUS // 2)))
    mode_text = font.render(mode_label, True, MUTED)
    screen.blit(mode_text, mode_text.get_rect(midright=(WIDTH - 14, STATUS // 2)))


def draw_modal(screen, font_big, font_small, lines, button_labels):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))
    box_w, box_h = WIDTH - 120, 100 + 30 * len(lines) + 70
    box = pygame.Rect((WIDTH - box_w) // 2, (HEIGHT - box_h) // 2, box_w, box_h)
    pygame.draw.rect(screen, (30, 36, 60), box, border_radius=16)
    pygame.draw.rect(screen, ACCENT, box, width=2, border_radius=16)
    y = box.top + 24
    for i, line in enumerate(lines):
        f = font_big if i == 0 else font_small
        text = f.render(line, True, TEXT)
        screen.blit(text, text.get_rect(centerx=box.centerx, top=y))
        y += 36 if i == 0 else 26
    buttons = []
    btn_w = 160
    gap = 18
    total = len(button_labels) * btn_w + (len(button_labels) - 1) * gap
    start_x = box.centerx - total // 2
    btn_y = box.bottom - 56
    for i, label in enumerate(button_labels):
        rect = pygame.Rect(start_x + i * (btn_w + gap), btn_y, btn_w, 40)
        pygame.draw.rect(screen, DROP_BTN, rect, border_radius=10)
        text = font_small.render(label, True, TEXT)
        screen.blit(text, text.get_rect(center=rect.center))
        buttons.append((label, rect))
    return buttons


def draw_endgame_card(screen, font_big, font_small, title, subtitle, button_labels,
                      glyph_color):
    margin_x = 20
    box = pygame.Rect(margin_x, 6, WIDTH - 2 * margin_x, STATUS + HEADER - 12)
    card = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
    pygame.draw.rect(card, (30, 36, 60, 235), card.get_rect(), border_radius=14)
    pygame.draw.rect(card, ACCENT, card.get_rect(), width=2, border_radius=14)
    screen.blit(card, box.topleft)

    glyph_cx = box.left + 24
    glyph_cy = box.top + 22
    pygame.draw.circle(screen, glyph_color, (glyph_cx, glyph_cy), 10)
    title_surf = font_big.render(title, True, TEXT)
    screen.blit(title_surf, title_surf.get_rect(midleft=(glyph_cx + 18, glyph_cy)))
    sub_surf = font_small.render(subtitle, True, MUTED)
    screen.blit(sub_surf, sub_surf.get_rect(midleft=(glyph_cx + 18, glyph_cy + 22)))

    buttons = []
    btn_w = 130
    btn_h = 34
    gap = 12
    total = len(button_labels) * btn_w + (len(button_labels) - 1) * gap
    start_x = box.right - total - 14
    btn_y = box.centery - btn_h // 2
    for i, label in enumerate(button_labels):
        rect = pygame.Rect(start_x + i * (btn_w + gap), btn_y, btn_w, btn_h)
        pygame.draw.rect(screen, DROP_BTN, rect, border_radius=8)
        text = font_small.render(label, True, TEXT)
        screen.blit(text, text.get_rect(center=rect.center))
        buttons.append((label, rect))
    return buttons


class StrategyWorker:

    def __init__(self, strategy_factory):
        self._factory = strategy_factory
        self._thread = None
        self._result = None
        self._lock = threading.Lock()

    def start(self, state):
        if self._thread is not None:
            return

        def run():
            strat = self._factory()
            decision = strat(state)
            with self._lock:
                self._result = decision

        self._result = None
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def poll(self):
        with self._lock:
            r = self._result
        if r is not None and self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self._result = None
            return r
        return None

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()


def make_mcts_factory(n_simulations, rollout="heuristic_win", tactical=False, seed=None):
    import random as _random
    from mcts import mcts_strategy
    def factory():
        rng = _random.Random(seed)
        return mcts_strategy(n_simulations=n_simulations, rollout=rollout,
                             tactical_root=tactical, rng=rng)
    return factory


def make_tree_factory(tree_path):
    import pickle as _pkl
    from decision_tree_builder import tree_strategy
    def factory():
        with open(tree_path, "rb") as f:
            tree = _pkl.load(f)
        return tree_strategy(tree)
    return factory


EASY = (100, "random", False)
MEDIUM = (400, "heuristic_win", True)
HARD = (800, "heuristic_win", True)


def _build_modes():
    modes = [
        ("Human vs Human", lambda: ("human", "human")),
        ("Human vs MCTS -- Easy", lambda: ("human", make_mcts_factory(*EASY))),
        ("Human vs MCTS -- Medium", lambda: ("human", make_mcts_factory(*MEDIUM))),
        ("Human vs MCTS -- Hard", lambda: ("human", make_mcts_factory(*HARD))),
        ("MCTS vs MCTS",
         lambda: (make_mcts_factory(*MEDIUM),
                  make_mcts_factory(*MEDIUM, seed=99))),
    ]
    tree_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_tree.pkl")
    if os.path.exists(tree_path):
        modes.extend([
            ("Human vs Tree",
             lambda: ("human", make_tree_factory(tree_path))),
            ("MCTS vs Tree",
             lambda: (make_mcts_factory(*MEDIUM), make_tree_factory(tree_path))),
        ])
    return modes


MODES = _build_modes()


def draw_menu(screen, font_big, font_small, mouse_pos):
    screen.fill((10, 14, 28))
    title = font_big.render("PopOut -- choose mode", True, TEXT)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 50)))

    n_modes = len(MODES)
    btn_h = 48
    gap = 10
    start_y = 100
    available = HEIGHT - start_y - 60
    if n_modes * btn_h + (n_modes - 1) * gap > available:
        btn_h = max(36, (available - (n_modes - 1) * gap) // n_modes)

    rects = []
    for i, (label, _) in enumerate(MODES):
        rect = pygame.Rect(WIDTH // 2 - 230, start_y + i * (btn_h + gap), 460, btn_h)
        is_hover = rect.collidepoint(mouse_pos)
        color = HOVER if is_hover else DROP_BTN
        pygame.draw.rect(screen, color, rect, border_radius=10)
        text = font_small.render(label, True, TEXT)
        screen.blit(text, text.get_rect(center=rect.center))
        rects.append((label, rect))

    hint = font_small.render(
        "ESC: back to menu  |  R: restart  |  close window: quit", True, MUTED,
    )
    screen.blit(hint, hint.get_rect(center=(WIDTH // 2, HEIGHT - 30)))
    return rects


def run():
    pygame.init()
    pygame.display.set_caption("PopOut -- IA 2025/2026")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Helvetica", 28, bold=True)
    font = pygame.font.SysFont("Helvetica", 22)
    font_small = pygame.font.SysFont("Helvetica", 18)

    in_menu = True
    state = initial_state()
    mode_label = ""
    p1_strat = p2_strat = None
    workers = {P1: None, P2: None}
    score = {P1: 0, P2: 0, "draw": 0}
    last_winner = None
    show_endgame_modal = False
    endgame_time = None
    WIN_REVEAL_DELAY = 1.5 

    def reset_game():
        nonlocal state, last_winner, show_endgame_modal, endgame_time
        state = initial_state()
        last_winner = None
        show_endgame_modal = False
        endgame_time = None

    def setup_mode(mode_idx):
        nonlocal p1_strat, p2_strat, mode_label, in_menu, workers
        label, factory_fn = MODES[mode_idx]
        mode_label = label
        p1_factory, p2_factory = factory_fn()
        p1_strat, p2_strat = p1_factory, p2_factory
        workers = {
            P1: None if p1_strat == "human" else StrategyWorker(p1_strat),
            P2: None if p2_strat == "human" else StrategyWorker(p2_strat),
        }
        reset_game()
        in_menu = False

    while True:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    in_menu = True
                    workers = {P1: None, P2: None}
                elif event.key == pygame.K_r and not in_menu:
                    reset_game()
            if in_menu and event.type == pygame.MOUSEBUTTONDOWN:
                rects = draw_menu(screen, font_big, font_small, mouse_pos)
                for i, (_, rect) in enumerate(rects):
                    if rect.collidepoint(event.pos):
                        setup_mode(i)
                        break
                continue
            if not in_menu and event.type == pygame.MOUSEBUTTONDOWN:
                if show_endgame_modal or state.winner is not None:
                    continue
                strat = p1_strat if state.player_to_move == P1 else p2_strat
                if strat != "human":
                    continue
                for c in range(COLS):
                    if drop_btn_rect(c).collidepoint(event.pos):
                        m = Move(c, "drop")
                        if m in legal_moves(state):
                            state = apply_move(state, m)
                            break
                else:
                    for c in range(COLS):
                        if pop_btn_rect(c).collidepoint(event.pos):
                            m = Move(c, "pop")
                            if m in legal_moves(state):
                                state = apply_move(state, m)
                                break

        if in_menu:
            draw_menu(screen, font_big, font_small, mouse_pos)
        else:
            if state.winner is None and not show_endgame_modal:
                worker = workers[state.player_to_move]
                if worker is not None:
                    decision = worker.poll()
                    if decision is not None:
                        if isinstance(decision, Move) and decision in legal_moves(state):
                            state = apply_move(state, decision)
                    elif not worker.busy:
                        worker.start(state)

            busy = any(w is not None and w.busy for w in workers.values())
            wcells = find_winning_cells(state) if state.winner else []
            draw_board(screen, font_small, state, mouse_pos, busy, winning_cells=wcells)
            draw_status(screen, font, state, mode_label, busy)

            if state.winner is not None and last_winner != state.winner:
                score[state.winner] += 1
                last_winner = state.winner
                endgame_time = time.time()

            if (state.winner is not None and not show_endgame_modal
                    and endgame_time is not None
                    and time.time() - endgame_time >= WIN_REVEAL_DELAY):
                show_endgame_modal = True

            if show_endgame_modal:
                title_line = "Draw" if state.winner == "draw" else f"P{state.winner} wins!"
                placar = (f"Score: P1 {score[P1]}  -  P2 {score[P2]}  "
                          f"(draws: {score['draw']})")
                if state.winner == "draw":
                    glyph_color = MUTED
                else:
                    glyph_color = P1_COLOR if state.winner == P1 else P2_COLOR
                buttons = draw_endgame_card(screen, font_big, font_small,
                                            title_line, placar,
                                            ["Play again", "Change mode"],
                                            glyph_color)
                if pygame.mouse.get_pressed()[0]:
                    px, py = pygame.mouse.get_pos()
                    for label, rect in buttons:
                        if rect.collidepoint(px, py):
                            if label == "Play again":
                                reset_game()
                            else:
                                in_menu = True
                                workers = {P1: None, P2: None}
                            time.sleep(0.15)  # debounce simples
                            break

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    sys.exit(run())
