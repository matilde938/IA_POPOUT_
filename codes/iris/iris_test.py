import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse

import numpy as np
import pandas as pd

from decision_tree_builder import (
    accuracy,
    fit_discretizer_equal_frequency,
    fit_discretizer_equal_width,
    fit_discretizer_supervised,
    id3,
    predict_batch,
    render_tree_matplotlib,
    render_tree_text,
    transform_discretizer,
    tree_size,
)

IRIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris.csv")


def load_iris():
    df = pd.read_csv(IRIS_PATH).drop(columns=["ID"])
    feats = [c for c in df.columns if c != "class"]
    return df[feats], df["class"], feats


def split(X, y, frac_train=0.8, seed=0):
    df = X.copy()
    df["__y"] = y.values
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    sp = int(frac_train * len(df))
    train, test = df.iloc[:sp], df.iloc[sp:]
    feats = [c for c in df.columns if c != "__y"]
    return train[feats], train["__y"], test[feats], test["__y"]


def confusion_matrix(y_true, y_pred, classes=None):
    if classes is None:
        classes = sorted(set(list(y_true) + list(y_pred)))
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(y_true, y_pred):
        M[idx[a], idx[b]] += 1
    return classes, M


def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


def evaluate(name, fit_fn, X_tr, y_tr, X_te, y_te, feats):
    fit = fit_fn(X_tr, y_tr, feats)
    Xtrd = transform_discretizer(X_tr, fit)
    Xted = transform_discretizer(X_te, fit)
    tree = id3(Xtrd, y_tr, feats)
    a_tr = accuracy(y_tr, predict_batch(tree, Xtrd))
    a_te = accuracy(y_te, predict_batch(tree, Xted))
    sz = tree_size(tree)
    return {
        "name": name, "tree": tree, "fit": fit,
        "acc_train": a_tr, "acc_test": a_te,
        "size": sz, "y_pred_test": predict_batch(tree, Xted),
    }


def kfold_eval(X, y, feats, fit_fn_factory, k=5, seed=0):
    accs, sizes = [], []
    for train_idx, test_idx in kfold_indices(len(X), k, seed=seed):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        fit = fit_fn_factory(X_tr, y_tr, feats)
        Xtrd = transform_discretizer(X_tr, fit)
        Xted = transform_discretizer(X_te, fit)
        tree = id3(Xtrd, y_tr, feats)
        accs.append(accuracy(y_te, predict_batch(tree, Xted)))
        sizes.append(tree_size(tree)["leaves"])
    return {
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_leaves": float(np.mean(sizes)), "accs": accs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../content")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=== Iris ===")
    X, y, feats = load_iris()
    print(f"Dataset: {len(X)} examples, classes: {sorted(y.unique())}\n")

    print("--- Discretisation strategies (split 80/20, seed=0) ---")
    print(f"{'strategy':<25} {'tr':>6} {'te':>6} {'leaves':>8} {'depth':>6}")
    X_tr, y_tr, X_te, y_te = split(X, y, seed=0)
    strategies = [
        ("supervised (binary)",
         lambda x, t, f: fit_discretizer_supervised(x, t, f)),
        ("equal_width(k=3)",
         lambda x, t, f: fit_discretizer_equal_width(x, f, n_bins=3)),
        ("equal_width(k=5)",
         lambda x, t, f: fit_discretizer_equal_width(x, f, n_bins=5)),
        ("equal_frequency(k=3)",
         lambda x, t, f: fit_discretizer_equal_frequency(x, f, n_bins=3)),
    ]
    results = [evaluate(n, fn, X_tr, y_tr, X_te, y_te, feats) for n, fn in strategies]
    for r in results:
        print(f"{r['name']:<25} {r['acc_train']:.3f}  {r['acc_test']:.3f}    "
              f"{r['size']['leaves']:>3}    {r['size']['depth']:>3}")
    print()

    if not args.quick:
        print("--- 5-fold cross-validation ---")
        print(f"{'strategy':<25} {'mean_acc':>10} {'std_acc':>10} {'mean_leaves':>13}")
        for name, fn in [
            ("supervised (binary)", lambda x, t, f: fit_discretizer_supervised(x, t, f)),
            ("equal_width(k=3)", lambda x, t, f: fit_discretizer_equal_width(x, f, n_bins=3)),
            ("equal_frequency(k=3)", lambda x, t, f: fit_discretizer_equal_frequency(x, f, n_bins=3)),
        ]:
            cv = kfold_eval(X, y, feats, fn, k=5, seed=0)
            print(f"{name:<25} {cv['mean_acc']:>10.3f} {cv['std_acc']:>10.3f}    "
                  f"{cv['mean_leaves']:>10.1f}")
        print()

    best = max(results, key=lambda r: r["acc_test"])
    print(f"Best strategy by test accuracy: {best['name']}\n")
    print("--- Confusion matrix ---")
    classes, M = confusion_matrix(y_te, best["y_pred_test"])
    print(f"classes = {classes}\n{M}\n")
    print("--- Tree (text) ---")
    print(render_tree_text(best["tree"]))
    print()

    # Guarda a arvore Iris (usada pelo notebook)
    import pickle
    iris_pkl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris_tree.pkl")
    with open(iris_pkl, "wb") as f:
        pickle.dump(best["tree"], f)
    print(f"[saved] {iris_pkl}")

    try:
        fig = render_tree_matplotlib(best["tree"], figsize=(14, 7))
        slug = (best["name"]
                .replace(" ", "_").replace("(", "").replace(")", "")
                .replace("=", "").replace(",", ""))
        out_png = os.path.join(out_dir, f"iris_tree_{slug}.png")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"[saved] {out_png}")
    except Exception as e:
        print(f"[warn] matplotlib failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())