#!/usr/bin/env python3
"""
PeerJ CS 141707 — corrected non-GNN structural baselines under Label v1.0.

Re-runs:
1) Node2Vec -> L2 logistic regression
2) 3-hop label propagation

New leakage-conscious implementation:
- train 2010-2018, validation 2019-2020, mature test 2021-2022
- frozen E1-E5 graph
- training-year edges only for structural representation
- topology-only union is symmetrized and duplicate node pairs collapsed
- Node2Vec: dim64, walk40, 10 walks/node, context10, p=q=1
- five genuinely independent embedding seeds
- label propagation: 3 hops, alpha=.9
- validation-only F1 threshold; test predictions persisted
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EXPECTED_HASHES = {
    "fraud_labels_v1_0.parquet": "1d50378138dfc65def76555c0a48aa9cea2656af4d456fe5178d9cf7eae31836",
    "node_features_v1_1.parquet": "adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2",
    "node_id_index_label_v1_0_fiverel_derived.parquet": "5602cf61f17d4b62598c3a13d98c72de728054c091cf01e31f2c3d6a067a2770",
    "global_edge_index.parquet": "dacdb3420917804bcbb9634ea83006d37f0b96a715cd990d241f706d7f066a09",
}
SEEDS = [42, 123, 456, 789, 1024]
TRAIN_YEARS = list(range(2010, 2019))
VAL_YEARS = [2019, 2020]
TEST_YEARS = [2021, 2022]

N2V_DIM = 64
N2V_WALK_LENGTH = 40
N2V_CONTEXT = 10
N2V_WALKS_PER_NODE = 10
N2V_NEGATIVE = 1
N2V_P = 1.0
N2V_Q = 1.0
LP_ALPHA = 0.9
LP_HOPS = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify_frozen_data(data_dir: Path):
    for fn, expected in EXPECTED_HASHES.items():
        p = data_dir / fn
        if not p.is_file():
            raise RuntimeError(f"MISSING_FROZEN_ARTIFACT:{p}")
        got = sha256(p)
        if got != expected:
            raise RuntimeError(f"HASH_MISMATCH:{fn}:expected={expected}:got={got}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_panel(data_dir: Path):
    labels = pd.read_parquet(
        data_dir / "fraud_labels_v1_0.parquet",
        columns=["company_code", "fiscal_year", "label_v1_strict_ab_primary"],
    ).rename(columns={"label_v1_strict_ab_primary": "target"})
    nf = pd.read_parquet(
        data_dir / "node_features_v1_1.parquet",
        columns=["firm_id", "year", "node_id"],
    ).copy()
    nf["company_code"] = nf["firm_id"].astype(str).str.zfill(6)
    df = nf.merge(
        labels,
        left_on=["company_code", "year"],
        right_on=["company_code", "fiscal_year"],
        how="left",
        validate="one_to_one",
    )
    if len(df) != 51675:
        raise RuntimeError(f"PANEL_ROWCOUNT_MISMATCH:{len(df)}")

    mapping = pd.read_parquet(
        data_dir / "node_id_index_label_v1_0_fiverel_derived.parquet",
        columns=["node_id", "node_idx"],
    ).copy()
    mapping["node_id"] = mapping["node_id"].astype(str)
    mapping["node_idx"] = mapping["node_idx"].astype(np.int64)
    n_total = len(mapping)
    idx = np.sort(mapping["node_idx"].to_numpy())
    if not np.array_equal(idx, np.arange(n_total, dtype=np.int64)):
        raise RuntimeError("NODE_MAPPING_NOT_DENSE_0_BASED")
    node_to_idx = dict(zip(mapping["node_id"], mapping["node_idx"]))
    df["node_idx"] = [node_to_idx[str(x)] for x in df["node_id"]]

    eligible = df["target"].notna().to_numpy()
    years = df["year"].to_numpy()
    y = df["target"].fillna(0).astype(int).to_numpy()
    tr = np.where(eligible & np.isin(years, TRAIN_YEARS))[0]
    va = np.where(eligible & np.isin(years, VAL_YEARS))[0]
    te = np.where(eligible & np.isin(years, TEST_YEARS))[0]
    assert (len(tr), int(y[tr].sum())) == (24106, 254)
    assert (len(va), int(y[va].sum())) == (7309, 40)
    assert (len(te), int(y[te].sum())) == (8435, 20)
    return df, y, tr, va, te, n_total


def build_training_union_graph(data_dir: Path, n_total: int):
    edges = pd.read_parquet(
        data_dir / "global_edge_index.parquet",
        columns=["src_idx", "dst_idx", "edge_type", "year"],
    )
    aliases = {
        "E1", "E1_HOLDS_BY", "E2", "E2_FLOAT_HELD_BY",
        "E3", "E3_HAS_MANAGER", "E4", "E4_CO_HELD", "E5", "E5_CO_MGR",
    }
    rels = set(edges["edge_type"].astype(str).unique())
    if not rels.issubset(aliases):
        raise RuntimeError(f"UNEXPECTED_RELATIONS:{sorted(rels-aliases)}")

    e = edges[edges["year"].isin(TRAIN_YEARS)][["src_idx", "dst_idx"]]
    src = e["src_idx"].to_numpy(np.int64)
    dst = e["dst_idx"].to_numpy(np.int64)
    if len(src) == 0:
        raise RuntimeError("NO_TRAINING_EDGES")

    # Symmetrize and collapse duplicate pairs. Relation identity/multiplicity ignored.
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    key = np.unique(a * np.int64(n_total) + b)
    u = (key // np.int64(n_total)).astype(np.int64)
    v = (key % np.int64(n_total)).astype(np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    active = np.unique(np.concatenate([u, v]))
    active_mask = np.zeros(n_total, dtype=bool)
    active_mask[active] = True
    return u, v, active, active_mask, len(src)


def import_core(code_dir: Path):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import formal_rerun_core as frc
    return frc


def node2vec_one_seed(
    seed, u, v, active, active_mask, n_total,
    df, y, tr, va, te, device, epochs, batch_size,
    outdir, code_dir,
):
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.nn.models import Node2Vec

    set_seed(seed)
    edge_index = torch.from_numpy(np.vstack([u, v]).astype(np.int64))
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=N2V_DIM,
        walk_length=N2V_WALK_LENGTH,
        context_size=N2V_CONTEXT,
        walks_per_node=N2V_WALKS_PER_NODE,
        p=N2V_P,
        q=N2V_Q,
        num_negative_samples=N2V_NEGATIVE,
        num_nodes=n_total,
        sparse=True,
    ).to(device)

    # Only active nodes are walk starts. num_workers=0 avoids GPU-model pickling.
    starts = torch.from_numpy(active.astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        starts.tolist(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=model.sample,
        generator=generator,
    )
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, nb, t0 = 0.0, 0, time.time()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            nb += 1
        mean_loss = total / max(nb, 1)
        history.append({"epoch": epoch, "mean_loss": mean_loss, "seconds": time.time()-t0})
        print(f"NODE2VEC seed={seed} epoch={epoch}/{epochs} loss={mean_loss:.6f}", flush=True)

    # Extract only company embeddings. Inactive companies are explicitly zeroed.
    company_nodes = np.unique(df["node_idx"].to_numpy(np.int64))
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(company_nodes).to(device)).detach().cpu().numpy().astype(np.float32)
    emb = {int(n): z[i] for i, n in enumerate(company_nodes)}
    X = np.zeros((len(df), N2V_DIM), dtype=np.float32)
    panel_nodes = df["node_idx"].to_numpy(np.int64)
    for i, n in enumerate(panel_nodes):
        if active_mask[n]:
            X[i] = emb[int(n)]

    clf = LogisticRegression(
        solver="liblinear", C=1.0, l1_ratio=0.0, class_weight=None,
        max_iter=5000, tol=1e-4, random_state=seed,
    )
    clf.fit(X[tr], y[tr])

    frc = import_core(code_dir)
    val_score = clf.predict_proba(X[va])[:, 1]
    val_auc = float(roc_auc_score(y[va], val_score))
    thr = frc.f1_val_threshold(y[va], val_score)
    test_score = clf.predict_proba(X[te])[:, 1]
    met = frc.metrics(y[te], test_score, thr)

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "company_code": df.iloc[te]["company_code"].to_numpy(),
        "fiscal_year": df.iloc[te]["year"].to_numpy(),
        "y_true": y[te],
        "score": test_score,
    }).to_parquet(outdir / "predictions.parquet", index=False)

    result = {
        "run_id": f"node2vec_seed{seed}",
        "model": "Node2Vec_L2Logistic",
        "seed": seed,
        "label_col": "label_v1_strict_ab_primary",
        "val_auc": val_auc,
        "val_threshold": float(thr),
        "metrics": met,
        "node2vec": {
            "embedding_dim": N2V_DIM, "walk_length": N2V_WALK_LENGTH,
            "context_size": N2V_CONTEXT, "walks_per_node": N2V_WALKS_PER_NODE,
            "p": N2V_P, "q": N2V_Q, "num_negative_samples": N2V_NEGATIVE,
            "epochs": epochs, "optimizer": "SparseAdam", "lr": 0.01,
            "independent_embedding_seed": True,
        },
        "downstream_classifier": {
            "type": "L2 logistic regression", "solver": "liblinear",
            "C": 1.0, "class_weight": None,
        },
        "graph_protocol": {
            "edge_years": TRAIN_YEARS, "future_edges_used": False,
            "symmetrized": True, "duplicate_pairs_collapsed": True,
            "relation_identity_used": False,
            "company_nodes_absent_from_training_graph": "zero embedding",
        },
        "embedding_history": history,
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def label_propagation(u, v, n_total, df, y, tr, va, te, outdir, code_dir):
    frc = import_core(code_dir)
    A = sparse.csr_matrix(
        (np.ones(len(u), dtype=np.float64), (u, v)),
        shape=(n_total, n_total),
    )
    degree = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(degree, dtype=np.float64)
    nz = degree > 0
    inv[nz] = 1.0 / degree[nz]
    P = sparse.diags(inv) @ A

    # Persistent company node gets the mean of its eligible training firm-year labels.
    tr_df = pd.DataFrame({
        "node_idx": df.iloc[tr]["node_idx"].to_numpy(np.int64),
        "target": y[tr].astype(np.float64),
    })
    seed_table = tr_df.groupby("node_idx", sort=False)["target"].mean()
    y0 = np.zeros(n_total, dtype=np.float64)
    y0[seed_table.index.to_numpy(np.int64)] = seed_table.to_numpy(np.float64)

    score = y0.copy()
    for _ in range(LP_HOPS):
        score = LP_ALPHA * (P @ score) + (1.0 - LP_ALPHA) * y0

    nodes = df["node_idx"].to_numpy(np.int64)
    val_score = score[nodes[va]]
    test_score = score[nodes[te]]
    val_auc = float(roc_auc_score(y[va], val_score))
    thr = frc.f1_val_threshold(y[va], val_score)
    met = frc.metrics(y[te], test_score, thr)

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "company_code": df.iloc[te]["company_code"].to_numpy(),
        "fiscal_year": df.iloc[te]["year"].to_numpy(),
        "y_true": y[te],
        "score": test_score,
    }).to_parquet(outdir / "predictions.parquet", index=False)

    result = {
        "run_id": "label_propagation_meanseed",
        "model": "LabelPropagation",
        "seed": None,
        "label_col": "label_v1_strict_ab_primary",
        "val_auc": val_auc,
        "val_threshold": float(thr),
        "metrics": met,
        "label_propagation": {
            "hops": LP_HOPS, "alpha": LP_ALPHA,
            "company_seed_rule": "mean eligible training firm-year target on persistent company node",
            "supporting_entity_seed": 0.0,
        },
        "graph_protocol": {
            "edge_years": TRAIN_YEARS, "future_edges_used": False,
            "symmetrized": True, "duplicate_pairs_collapsed": True,
            "relation_identity_used": False,
        },
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def bootstrap_structural(structural_root: Path, meta: Path, n_boot: int):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

    def p5(y, s, w):
        k = max(1.0, math.ceil(float(w.sum()) * 0.05))
        order = np.argsort(-s, kind="mergesort")
        yy, ww = y[order], w[order].astype(float)
        cs = np.cumsum(ww)
        before = np.concatenate(([0.0], cs[:-1]))
        take = np.clip(k - before, 0.0, ww)
        return float(np.sum(take * yy) / take.sum())

    def r10(y, s, w):
        fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
        ok = np.where(fpr <= 0.10 + 1e-12)[0]
        return float(np.max(tpr[ok])) if len(ok) else 0.0

    def wm(y, s, w, thr):
        return np.array([
            roc_auc_score(y, s, sample_weight=w),
            average_precision_score(y, s, sample_weight=w),
            f1_score(y, (s >= thr).astype(int), sample_weight=w, zero_division=0),
            p5(y, s, w),
            r10(y, s, w),
        ], dtype=float)

    n2v = []
    for seed in SEEDS:
        od = structural_root / f"node2vec_seed{seed}"
        j = json.loads((od / "result.json").read_text())
        p = pd.read_parquet(od / "predictions.parquet").sort_values(
            ["company_code", "fiscal_year"]
        ).reset_index(drop=True)
        n2v.append((j, p))

    lp_od = structural_root / "label_propagation"
    lp_j = json.loads((lp_od / "result.json").read_text())
    lp = pd.read_parquet(lp_od / "predictions.parquet").sort_values(
        ["company_code", "fiscal_year"]
    ).reset_index(drop=True)

    master = n2v[0][1]
    y = master["y_true"].to_numpy(int)
    assert (len(y), int(y.sum())) == (8435, 20)
    keys = list(zip(master["company_code"].astype(str), master["fiscal_year"].astype(int)))

    for _, p in n2v[1:]:
        if list(zip(p["company_code"].astype(str), p["fiscal_year"].astype(int))) != keys:
            raise RuntimeError("N2V_KEY_MISMATCH")
    if list(zip(lp["company_code"].astype(str), lp["fiscal_year"].astype(int))) != keys:
        raise RuntimeError("LP_KEY_MISMATCH")

    cat = pd.Categorical(master["company_code"].astype(str))
    cluster_idx = cat.codes.astype(np.int32)
    n_clusters = len(cat.categories)
    n2v_scores = [p["score"].to_numpy(float) for _, p in n2v]
    n2v_thr = [float(j["val_threshold"]) for j, _ in n2v]
    lp_score = lp["score"].to_numpy(float)
    lp_thr = float(lp_j["val_threshold"])
    one = np.ones(len(y), dtype=float)

    obs_n2v = np.mean([wm(y, s, one, t) for s, t in zip(n2v_scores, n2v_thr)], axis=0)
    obs_lp = wm(y, lp_score, one, lp_thr)
    bn = np.empty((n_boot, 5), dtype=float)
    bl = np.empty((n_boot, 5), dtype=float)
    rng = np.random.default_rng(141707)

    for b in range(n_boot):
        while True:
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            counts = np.bincount(sampled, minlength=n_clusters)
            w = counts[cluster_idx].astype(float)
            if np.sum(w*y) > 0 and np.sum(w*(1-y)) > 0:
                break
        bn[b] = np.mean([wm(y, s, w, t) for s, t in zip(n2v_scores, n2v_thr)], axis=0)
        bl[b] = wm(y, lp_score, w, lp_thr)
        if b == 0 or (b + 1) % 100 == 0:
            print(f"bootstrap {b+1}/{n_boot}", flush=True)

    names = ["roc_auc", "ap", "f1_val_threshold", "p_at_5pct", "r_at_10fpr"]
    rows = []
    for condition, obs, arr, nruns in [
        ("Node2Vec_L2Logistic", obs_n2v, bn, 5),
        ("LabelPropagation", obs_lp, bl, 1),
    ]:
        row = {
            "condition": condition, "n_seeds_or_runs": nruns, "n_boot": n_boot,
            "cluster_unit": "company_code", "test_n": len(y), "test_pos": int(y.sum()),
            "company_clusters": n_clusters,
        }
        for i, name in enumerate(names):
            row[name + "_observed"] = float(obs[i])
            row[name + "_ci_lo"] = float(np.quantile(arr[:, i], .025))
            row[name + "_ci_hi"] = float(np.quantile(arr[:, i], .975))
        rows.append(row)
    pd.DataFrame(rows).to_csv(meta / "structural_company_cluster_bootstrap_ci.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "cyq/peerj_141707_rerun_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--node2vec-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    root = args.root.resolve()
    allowed = (Path.home() / "cyq").resolve()
    if allowed != root and allowed not in root.parents:
        raise RuntimeError("REFUSE_WRITE_OUTSIDE_HOME_CYQ")

    data_dir = root / "data"
    code_dir = root / "code"
    structural_root = root / "results/structural_baselines_v1"
    meta = root / "meta/structural_baselines_v1"
    structural_root.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    print("PeerJ 141707 | structural baselines v1", flush=True)
    print("VERIFY frozen artifact hashes...", flush=True)
    verify_frozen_data(data_dir)

    try:
        import torch, torch_geometric, torch_cluster
    except Exception as e:
        raise RuntimeError(
            "NODE2VEC_BACKEND_MISSING: install torch_cluster matching Torch 2.11 + cu128"
        ) from e

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")

    df, y, tr, va, te, n_total = load_panel(data_dir)

    print("BUILD training-only E1-E5 union graph...", flush=True)
    u, v, active, active_mask, raw_train_edges = build_training_union_graph(data_dir, n_total)
    graph_manifest = {
        "n_total_nodes": n_total,
        "raw_training_edge_rows": raw_train_edges,
        "symmetrized_deduplicated_edges": int(len(u)),
        "active_nodes": int(len(active)),
        "edge_years": TRAIN_YEARS,
        "future_edges_used": False,
        "symmetrized": True,
        "duplicate_pairs_collapsed": True,
        "relation_identity_used": False,
        "test_n": len(te),
        "test_pos": int(y[te].sum()),
    }
    (meta / "graph_manifest.json").write_text(json.dumps(graph_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(graph_manifest, indent=2, sort_keys=True), flush=True)

    lp_dir = structural_root / "label_propagation"
    if not (lp_dir / "result.json").is_file():
        print("RUN label propagation...", flush=True)
        lp = label_propagation(u, v, n_total, df, y, tr, va, te, lp_dir, code_dir)
        print(f"DONE label propagation test_auc={lp['metrics']['roc_auc']:.6f}", flush=True)
    else:
        print("SKIP completed label propagation", flush=True)

    for seed in SEEDS:
        od = structural_root / f"node2vec_seed{seed}"
        if (od / "result.json").is_file():
            print(f"SKIP completed Node2Vec seed={seed}", flush=True)
            continue
        print(f"RUN Node2Vec seed={seed}", flush=True)
        res = node2vec_one_seed(
            seed, u, v, active, active_mask, n_total,
            df, y, tr, va, te, args.device, args.node2vec_epochs,
            args.batch_size, od, code_dir,
        )
        print(
            f"DONE Node2Vec seed={seed} val_auc={res['val_auc']:.6f} "
            f"test_auc={res['metrics']['roc_auc']:.6f}",
            flush=True,
        )

    files = sorted(structural_root.glob("node2vec_seed*/result.json"))
    if len(files) != 5:
        raise RuntimeError(f"NODE2VEC_RESULT_COUNT:{len(files)}")

    rows = []
    for p in files:
        j = json.loads(p.read_text())
        m = j["metrics"]
        rows.append({
            "model": "Node2Vec_L2Logistic", "seed": j["seed"], "val_auc": j["val_auc"],
            "roc_auc": m["roc_auc"], "ap": m["ap"],
            "f1_val_threshold": m["f1_val_threshold"],
            "p_at_5pct": m["p_at_5pct"], "r_at_10fpr": m["r_at_10fpr"],
        })
    per_seed = pd.DataFrame(rows).sort_values("seed")
    per_seed.to_csv(structural_root / "node2vec_summary.per_seed.csv", index=False)

    n2v_summary = {"model": "Node2Vec_L2Logistic", "n_independent_embeddings": 5}
    for c in ["roc_auc", "ap", "f1_val_threshold", "p_at_5pct", "r_at_10fpr"]:
        n2v_summary[c + "_mean"] = float(per_seed[c].mean())
        n2v_summary[c + "_sd"] = float(per_seed[c].std(ddof=1))

    lp_j = json.loads((lp_dir / "result.json").read_text())
    lm = lp_j["metrics"]
    pd.DataFrame([
        n2v_summary,
        {
            "model": "LabelPropagation", "n_independent_embeddings": 1,
            "roc_auc_mean": lm["roc_auc"], "roc_auc_sd": 0.0,
            "ap_mean": lm["ap"], "ap_sd": 0.0,
            "f1_val_threshold_mean": lm["f1_val_threshold"], "f1_val_threshold_sd": 0.0,
            "p_at_5pct_mean": lm["p_at_5pct"], "p_at_5pct_sd": 0.0,
            "r_at_10fpr_mean": lm["r_at_10fpr"], "r_at_10fpr_sd": 0.0,
        }
    ]).to_csv(structural_root / "structural_baselines_v1_summary.csv", index=False)

    print("RUN company-cluster bootstrap...", flush=True)
    bootstrap_structural(structural_root, meta, args.n_boot)

    report = [
        "PeerJ 141707 | Structural baselines v1",
        "label=label_v1_strict_ab_primary",
        "train=2010-2018", "validation=2019-2020", "test=2021-2022",
        "test_n=8435", "test_pos=20",
        "graph_edges=training-years-only E1-E5 union",
        "future_edges_used=false",
        "symmetrized=true",
        "duplicate_pairs_collapsed=true",
        "relation_identity_used=false",
        "node2vec_independent_embeddings=5",
        "node2vec_dim=64", "node2vec_walk_length=40",
        "node2vec_walks_per_node=10", "node2vec_context_size=10",
        "node2vec_p=1.0", "node2vec_q=1.0",
        f"node2vec_epochs={args.node2vec_epochs}",
        "label_propagation_hops=3", "label_propagation_alpha=0.9",
        "label_propagation_company_seed_rule=training firm-year mean",
        f"bootstrap_n={args.n_boot}", "bootstrap_cluster=company_code",
        "STATUS: STRUCTURAL_BASELINES_V1_COMPLETE",
    ]
    (meta / "structural_baselines_v1_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
