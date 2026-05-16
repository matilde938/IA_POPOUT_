
import math
import re
from collections import Counter

import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature=None, children=None, label=None, n_samples=0, class_counts=None):
        self.feature = feature
        self.children = children if children is not None else {}
        self.label = label
        self.n_samples = n_samples
        self.class_counts = class_counts if class_counts is not None else {}

    @property
    def is_leaf(self):
        return self.label is not None

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(label={self.label!r}, n={self.n_samples})"
        return f"Node(feature={self.feature!r}, |children|={len(self.children)})"

def entropy(y):
    if len(y) == 0:
        return 0.0
    counts = Counter(y)
    total = len(y)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def information_gain(X, y, feature):
    base = entropy(y)
    total = len(y)
    if total == 0:
        return 0.0
    cond = 0.0
    for value in X[feature].unique():
        sub_y = y[X[feature] == value]
        if len(sub_y):
            cond += (len(sub_y) / total) * entropy(sub_y)
    return base - cond

def id3(X, y, features, parent_majority=None, max_depth=None, depth=0):
    counts = dict(Counter(y))
    n_samples = len(y)

    if n_samples == 0:
        return Node(label=parent_majority, n_samples=0)

    if len(counts) == 1:
        only_class = next(iter(counts))
        return Node(label=only_class, n_samples=n_samples, class_counts=counts)

    if not features or (max_depth is not None and depth >= max_depth):
        majority = max(counts.items(), key=lambda kv: kv[1])[0]
        return Node(label=majority, n_samples=n_samples, class_counts=counts)

    gains = [(f, information_gain(X, y, f)) for f in features]
    best_feature, best_gain = max(gains, key=lambda x: x[1])

    if best_gain <= 0:
        majority = max(counts.items(), key=lambda kv: kv[1])[0]
        return Node(label=majority, n_samples=n_samples, class_counts=counts)

    node = Node(feature=best_feature, n_samples=n_samples, class_counts=counts)
    majority = max(counts.items(), key=lambda kv: kv[1])[0]
    remaining = [f for f in features if f != best_feature]

    for value in sorted(X[best_feature].unique(), key=str):
        mask = X[best_feature] == value
        node.children[value] = id3(
            X[mask], y[mask], remaining,
            parent_majority=majority,
            max_depth=max_depth,
            depth=depth + 1,
        )
    return node


def predict(tree, sample):
    node = tree
    while not node.is_leaf:
        value = sample.get(node.feature)
        if value in node.children:
            node = node.children[value]
        else:
            return _majority_label(node)
    return node.label


def _majority_label(node):
    if node.is_leaf:
        return node.label
    counts = Counter()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            counts[n.label] += n.n_samples
        else:
            stack.extend(n.children.values())
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def predict_batch(tree, X):
    return [predict(tree, row.to_dict()) for _, row in X.iterrows()]


def accuracy(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

class DiscretizationFit:
    def __init__(self, strategy, edges, labels):
        self.strategy = strategy
        self.edges = edges
        self.labels = labels


def fit_discretizer_equal_width(X, columns, n_bins=3):
    edges, labels = {}, {}
    for col in columns:
        col_min, col_max = float(X[col].min()), float(X[col].max())
        if col_min == col_max:
            edges[col] = np.array([col_min, col_max])
            labels[col] = ["bin0"]
            continue
        edges[col] = np.linspace(col_min, col_max, n_bins + 1)
        labels[col] = [f"bin{i}" for i in range(n_bins)]
    return DiscretizationFit("equal_width", edges, labels)


def fit_discretizer_equal_frequency(X, columns, n_bins=3):
    edges, labels = {}, {}
    for col in columns:
        quantiles = np.linspace(0, 1, n_bins + 1)
        e = np.unique(np.quantile(X[col].values, quantiles))
        edges[col] = e
        labels[col] = [f"bin{i}" for i in range(len(e) - 1)]
    return DiscretizationFit("equal_frequency", edges, labels)


def fit_discretizer_supervised(X, y, columns):
    edges, labels = {}, {}
    for col in columns:
        values = X[col].values
        sorted_vals = np.sort(np.unique(values))
        candidates = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        base_h = entropy(y)
        n = len(y)
        best_t = None
        best_gain = -1.0
        for t in candidates:
            left, right = y[values <= t], y[values > t]
            if len(left) == 0 or len(right) == 0:
                continue
            cond = (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)
            gain = base_h - cond
            if gain > best_gain:
                best_gain = gain
                best_t = float(t)
        if best_t is None:
            best_t = float(np.median(values))
        edges[col] = np.array([-np.inf, best_t, np.inf])
        labels[col] = ["low", "high"]
    return DiscretizationFit("supervised", edges, labels)


def transform_discretizer(X, fit):
    out = X.copy()
    for col, e in fit.edges.items():
        if col not in out.columns:
            continue
        bins = np.searchsorted(e, out[col].values, side="right") - 1
        bins = np.clip(bins, 0, len(fit.labels[col]) - 1)
        out[col] = [fit.labels[col][i] for i in bins]
    return out


def render_tree_text(tree):
    lines = []
    _render_text_recursive(tree, lines, prefix="", value_label=None)
    return "\n".join(lines)


def _render_text_recursive(node, lines, prefix, value_label):
    head = f"{prefix}└─ [{value_label}] " if value_label is not None else ""
    if node.is_leaf:
        counts_str = " " + str(dict(node.class_counts)) if node.class_counts else ""
        lines.append(f"{head}class: {node.label} (n={node.n_samples}){counts_str}")
        return
    lines.append(f"{head}{node.feature}? (n={node.n_samples})")
    next_prefix = prefix + "   "
    for val, child in node.children.items():
        _render_text_recursive(child, lines, prefix=next_prefix, value_label=val)


def render_tree_matplotlib(tree, figsize=(12, 6)):
    import matplotlib.pyplot as plt

    positions = {}
    leaves = []
    _collect_leaves(tree, leaves)
    n_leaves = max(1, len(leaves))

    leaf_idx = [0]

    def assign(node, depth):
        if node.is_leaf:
            x = leaf_idx[0] / max(1, n_leaves - 1)
            leaf_idx[0] += 1
            positions[id(node)] = (x, -depth)
            return x
        xs = [assign(c, depth + 1) for c in node.children.values()]
        x = sum(xs) / len(xs)
        positions[id(node)] = (x, -depth)
        return x

    assign(tree, 0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    def draw(node, parent_pos=None, edge_label=None):
        x, y = positions[id(node)]
        if parent_pos is not None:
            px, py = parent_pos
            ax.plot([px, x], [py, y], "k-", linewidth=1, zorder=1)
            if edge_label is not None:
                ax.text((px + x) / 2, (py + y) / 2, str(edge_label),
                        fontsize=8, ha="center",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", ec="gray", lw=0.5))
        if node.is_leaf:
            ax.text(x, y, f"{node.label}\n(n={node.n_samples})",
                    fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#dff2d8", ec="#5fa84a"))
        else:
            ax.text(x, y, f"{node.feature}?",
                    fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#cee0f0", ec="#3b6da3"))
            for val, child in node.children.items():
                draw(child, parent_pos=(x, y), edge_label=val)

    draw(tree)
    ax.margins(x=0.10, y=0.18)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.06)
    return fig


def _collect_leaves(node, out):
    if node.is_leaf:
        out.append(node)
    else:
        for c in node.children.values():
            _collect_leaves(c, out)


def tree_size(tree):
    leaves = []
    _collect_leaves(tree, leaves)
    return {
        "nodes": _count_nodes(tree),
        "leaves": len(leaves),
        "depth": _max_depth(tree),
    }


def _count_nodes(node):
    if node.is_leaf:
        return 1
    return 1 + sum(_count_nodes(c) for c in node.children.values())


def _max_depth(node):
    if node.is_leaf:
        return 0
    return 1 + max(_max_depth(c) for c in node.children.values())

_MOVE_RE = re.compile(r"^([dp])([0-6])$")


def encode_state_for_tree(state):
    flat = state.board.reshape(-1).tolist()
    features = {f"s{i}": int(flat[i]) for i in range(42)}
    features["to_play"] = int(state.player_to_move)
    return features


def decode_move_string(move_str):
    from popout import Move
    if not isinstance(move_str, str):
        return None
    m = _MOVE_RE.match(move_str)
    if not m:
        return None
    return Move(int(m.group(2)), "drop" if m.group(1) == "d" else "pop")


def _fallback_legal_move(state, predicted):
    from popout import COLS, legal_moves
    legal = legal_moves(state)
    if not legal:
        raise RuntimeError("No legal moves in non-terminal state.")
    if predicted is not None:
        same_kind = [m for m in legal if m.kind == predicted.kind]
        if same_kind:
            same_kind.sort(key=lambda m: abs(m.column - COLS // 2))
            return same_kind[0]
    drops = [m for m in legal if m.kind == "drop"]
    if drops:
        drops.sort(key=lambda m: abs(m.column - COLS // 2))
        return drops[0]
    return legal[0]


def tree_strategy(tree):
    from popout import legal_moves

    def strat(state):
        features = encode_state_for_tree(state)
        move_str = predict(tree, features)
        predicted = decode_move_string(move_str) if move_str is not None else None
        if predicted is not None and predicted in legal_moves(state):
            return predicted
        return _fallback_legal_move(state, predicted)

    return strat
