#!/usr/bin/env python3
"""End-to-end Elliptic fraud detection analysis pipeline.

Covers:
- Data understanding (class distribution, label availability over time, graph stats)
- Feature-only baselines (LogReg, RandomForest, XGBoost)
- GNN models (GCN, GraphSAGE, GAT) when torch-geometric is available
- Evaluation with illicit-class precision/recall/F1/PR-AUC and post-shutdown drop analysis
- Neighbor influence analysis for GAT via attention weights
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


UNKNOWN_LABEL = -1
ILLICIT_LABEL = 1
LICIT_LABEL = 0


@dataclass
class ExperimentConfig:
    train_end_step: int = 34
    val_end_step: int = 43
    illicit_label: int = ILLICIT_LABEL
    random_state: int = 42
    gnn_epochs: int = 100
    gnn_lr: float = 0.01
    gnn_hidden_dim: int = 64


def temporal_split_indices(
    time_steps: Sequence[int],
    train_end_step: int,
    val_end_step: int,
) -> Dict[str, List[int]]:
    train_idx = [i for i, t in enumerate(time_steps) if t <= train_end_step]
    val_idx = [i for i, t in enumerate(time_steps) if train_end_step < t <= val_end_step]
    test_idx = [i for i, t in enumerate(time_steps) if t > val_end_step]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def illicit_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
    illicit_label: int = ILLICIT_LABEL,
) -> Dict[str, float]:
    pairs = [(int(t == illicit_label), float(s)) for t, s in zip(y_true, y_score)]
    positives = sum(t for t, _ in pairs)

    y_pred = [1 if s >= threshold else 0 for _, s in pairs]
    tp = sum(1 for (t, _), p in zip(pairs, y_pred) if t == 1 and p == 1)
    fp = sum(1 for (t, _), p in zip(pairs, y_pred) if t == 0 and p == 1)
    fn = sum(1 for (t, _), p in zip(pairs, y_pred) if t == 1 and p == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    if positives == 0:
        pr_auc = 0.0
    else:
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        pr_points: List[Tuple[float, float]] = [(0.0, 1.0)]
        running_tp = 0
        running_fp = 0
        for true_label, _ in sorted_pairs:
            if true_label == 1:
                running_tp += 1
            else:
                running_fp += 1
            p = _safe_div(running_tp, running_tp + running_fp)
            r = _safe_div(running_tp, positives)
            pr_points.append((r, p))

        pr_auc = 0.0
        for (r1, p1), (r2, p2) in zip(pr_points[:-1], pr_points[1:]):
            pr_auc += (r2 - r1) * (p1 + p2) / 2

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
    }


def load_elliptic_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import pandas as pd

    features_path = data_dir / "elliptic_txs_features.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)

    renamed_cols = ["tx_id", "time_step"] + [f"f_{i}" for i in range(features.shape[1] - 2)]
    features.columns = renamed_cols

    classes = classes.rename(columns={"txId": "tx_id", "class": "raw_class"})
    label_map = {"1": ILLICIT_LABEL, "2": LICIT_LABEL, "unknown": UNKNOWN_LABEL, 1: ILLICIT_LABEL, 2: LICIT_LABEL}
    classes["label"] = classes["raw_class"].map(label_map).fillna(UNKNOWN_LABEL).astype(int)

    edges = edges.rename(columns={"txId1": "src", "txId2": "dst"})

    return features, classes[["tx_id", "label"]], edges


def data_understanding(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
) -> Dict[str, Any]:
    merged = features[["tx_id", "time_step"]].merge(classes, on="tx_id", how="left")
    merged["label"] = merged["label"].fillna(UNKNOWN_LABEL).astype(int)

    class_counts = merged["label"].value_counts().to_dict()
    class_distribution = {
        "illicit": int(class_counts.get(ILLICIT_LABEL, 0)),
        "licit": int(class_counts.get(LICIT_LABEL, 0)),
        "unknown": int(class_counts.get(UNKNOWN_LABEL, 0)),
    }

    by_time = (
        merged.groupby("time_step")
        .agg(
            total=("tx_id", "count"),
            labeled=("label", lambda s: int((s != UNKNOWN_LABEL).sum())),
            illicit=("label", lambda s: int((s == ILLICIT_LABEL).sum())),
            licit=("label", lambda s: int((s == LICIT_LABEL).sum())),
        )
        .reset_index()
    )
    by_time["label_availability"] = by_time["labeled"] / by_time["total"]

    node_count = features["tx_id"].nunique()
    edge_count = len(edges)
    degrees: Dict[Any, int] = {}
    for row in edges.itertuples(index=False):
        degrees[row.src] = degrees.get(row.src, 0) + 1
        degrees[row.dst] = degrees.get(row.dst, 0) + 1
    isolated = int(node_count - len(degrees))

    graph_stats = {
        "nodes": int(node_count),
        "edges": int(edge_count),
        "avg_degree": _safe_div(2 * edge_count, node_count),
        "density_undirected": _safe_div(2 * edge_count, node_count * (node_count - 1)),
        "isolated_nodes": isolated,
    }

    return {
        "class_distribution": class_distribution,
        "label_availability_by_time": by_time.to_dict(orient="records"),
        "graph_structure": graph_stats,
    }


def _prepare_feature_table(
    features: pd.DataFrame,
    classes: pd.DataFrame,
) -> pd.DataFrame:
    full = features.merge(classes, on="tx_id", how="left")
    full["label"] = full["label"].fillna(UNKNOWN_LABEL).astype(int)
    return full


def run_feature_baselines(full: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    labeled = full[full["label"] != UNKNOWN_LABEL].copy()
    splits = temporal_split_indices(
        labeled["time_step"].tolist(),
        train_end_step=config.train_end_step,
        val_end_step=config.val_end_step,
    )

    feature_cols = [c for c in labeled.columns if c.startswith("f_")]
    X = labeled[feature_cols]
    y = labeled["label"]

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    if not train_idx or not test_idx:
        raise ValueError("Temporal split produced empty train or test set; check time steps/config.")

    model_specs: Dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        model_specs["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=config.random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    results: Dict[str, Any] = {"models": {}, "time_split": {"train_end": config.train_end_step, "val_end": config.val_end_step}}

    eval_non_train = labeled.index.difference(labeled.iloc[train_idx].index)

    for model_name, model in model_specs.items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        test_scores = model.predict_proba(X.iloc[test_idx])[:, 1]
        test_metrics = illicit_metrics(y.iloc[test_idx].tolist(), test_scores.tolist(), illicit_label=config.illicit_label)

        eval_slice = labeled.loc[eval_non_train]
        eval_scores = model.predict_proba(eval_slice[feature_cols])[:, 1]
        pre_mask = eval_slice["time_step"] <= config.val_end_step
        post_mask = eval_slice["time_step"] > config.val_end_step

        pre_metrics = illicit_metrics(
            eval_slice.loc[pre_mask, "label"].tolist(),
            eval_scores[pre_mask.to_numpy()].tolist(),
            illicit_label=config.illicit_label,
        ) if int(pre_mask.sum()) else None
        post_metrics = illicit_metrics(
            eval_slice.loc[post_mask, "label"].tolist(),
            eval_scores[post_mask.to_numpy()].tolist(),
            illicit_label=config.illicit_label,
        ) if int(post_mask.sum()) else None

        results["models"][model_name] = {
            "test_metrics": test_metrics,
            "shutdown_analysis": {
                "at_or_before_step_43": pre_metrics,
                "after_step_43": post_metrics,
            },
            "samples": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
        }

    return results


def run_gnn_models(
    full: pd.DataFrame,
    edges: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv

    tx_ids = full["tx_id"].tolist()
    id_to_idx = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    edge_index_rows = [(id_to_idx[s], id_to_idx[d]) for s, d in edges[["src", "dst"]].itertuples(index=False) if s in id_to_idx and d in id_to_idx]
    if not edge_index_rows:
        raise ValueError("No usable edges found after node ID alignment.")

    edge_index = torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous()
    x_cols = [c for c in full.columns if c.startswith("f_")]
    x = torch.tensor(full[x_cols].to_numpy(), dtype=torch.float32)
    y = torch.tensor(full["label"].to_numpy(), dtype=torch.long)
    t = full["time_step"].to_numpy()

    labeled_mask = y != UNKNOWN_LABEL
    train_mask = torch.tensor((t <= config.train_end_step) & (full["label"].to_numpy() != UNKNOWN_LABEL), dtype=torch.bool)
    val_mask = torch.tensor((t > config.train_end_step) & (t <= config.val_end_step) & (full["label"].to_numpy() != UNKNOWN_LABEL), dtype=torch.bool)
    test_mask = torch.tensor((t > config.val_end_step) & (full["label"].to_numpy() != UNKNOWN_LABEL), dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, y=y)

    class GCN(nn.Module):
        def __init__(self, in_dim: int, hidden: int):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, 1)

        def forward(self, d: Data) -> torch.Tensor:
            h = F.relu(self.conv1(d.x, d.edge_index))
            h = F.dropout(h, p=0.2, training=self.training)
            return self.conv2(h, d.edge_index).squeeze(-1)

    class GraphSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden: int):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, 1)

        def forward(self, d: Data) -> torch.Tensor:
            h = F.relu(self.conv1(d.x, d.edge_index))
            h = F.dropout(h, p=0.2, training=self.training)
            return self.conv2(h, d.edge_index).squeeze(-1)

    class GAT(nn.Module):
        def __init__(self, in_dim: int, hidden: int):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden, heads=4, concat=False, dropout=0.2)
            self.conv2 = GATConv(hidden, 1, heads=1, concat=False, dropout=0.2)

        def forward(self, d: Data) -> torch.Tensor:
            h = F.elu(self.conv1(d.x, d.edge_index))
            h = F.dropout(h, p=0.2, training=self.training)
            return self.conv2(h, d.edge_index).squeeze(-1)

        def attention(self, d: Data):
            _, (edge_idx, alpha) = self.conv1(d.x, d.edge_index, return_attention_weights=True)
            return edge_idx, alpha

    model_builders = {
        "gcn": GCN,
        "graphsage": GraphSAGE,
        "gat": GAT,
    }

    results: Dict[str, Any] = {"models": {}}

    target = (y == config.illicit_label).float()

    for name, builder in model_builders.items():
        model = builder(x.shape[1], config.gnn_hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.gnn_lr)

        for _ in range(config.gnn_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data)
            loss = F.binary_cross_entropy_with_logits(logits[train_mask], target[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores_all = torch.sigmoid(model(data)).cpu()

        test_scores = scores_all[test_mask].numpy().tolist()
        test_labels = y[test_mask].cpu().numpy().tolist()
        test_metrics = illicit_metrics(test_labels, test_scores, illicit_label=config.illicit_label)

        eval_mask = labeled_mask & (~train_mask)
        eval_times = t[eval_mask.cpu().numpy()]
        eval_labels = y[eval_mask].cpu().numpy().tolist()
        eval_scores = scores_all[eval_mask].numpy().tolist()

        pre_idx = [i for i, step in enumerate(eval_times) if step <= config.val_end_step]
        post_idx = [i for i, step in enumerate(eval_times) if step > config.val_end_step]

        pre_metrics = illicit_metrics([eval_labels[i] for i in pre_idx], [eval_scores[i] for i in pre_idx], illicit_label=config.illicit_label) if pre_idx else None
        post_metrics = illicit_metrics([eval_labels[i] for i in post_idx], [eval_scores[i] for i in post_idx], illicit_label=config.illicit_label) if post_idx else None

        model_result: Dict[str, Any] = {
            "test_metrics": test_metrics,
            "shutdown_analysis": {
                "at_or_before_step_43": pre_metrics,
                "after_step_43": post_metrics,
            },
        }

        if name == "gat":
            edge_idx, alpha = model.attention(data)
            alpha_vec = alpha.mean(dim=1) if alpha.ndim > 1 else alpha
            top_illicit_nodes = sorted(
                [
                    (i, score)
                    for i, (score, label) in enumerate(zip(scores_all.tolist(), y.tolist()))
                    if label != UNKNOWN_LABEL and score >= 0.5
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            neighbor_explanations: Dict[str, List[Dict[str, float]]] = {}
            src = edge_idx[0].cpu().tolist()
            dst = edge_idx[1].cpu().tolist()
            weights = alpha_vec.cpu().tolist()

            for node_idx, _ in top_illicit_nodes:
                incoming = [
                    (src_i, wt)
                    for src_i, dst_i, wt in zip(src, dst, weights)
                    if dst_i == node_idx
                ]
                incoming.sort(key=lambda z: z[1], reverse=True)
                top_neighbors = [
                    {"neighbor_tx_id": int(tx_ids[n_idx]), "attention_weight": float(wt)}
                    for n_idx, wt in incoming[:5]
                ]
                neighbor_explanations[str(int(tx_ids[node_idx]))] = top_neighbors

            model_result["neighbor_analysis"] = neighbor_explanations

        results["models"][name] = model_result

    return results


def analyze_graph_value(baseline_results: Dict[str, Any], gnn_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_scores = {
        name: stats["test_metrics"]["f1"]
        for name, stats in baseline_results.get("models", {}).items()
    }
    best_baseline_name = max(baseline_scores, key=baseline_scores.get) if baseline_scores else None
    best_baseline_f1 = baseline_scores.get(best_baseline_name) if best_baseline_name else None

    if not gnn_results or not gnn_results.get("models"):
        return {
            "graph_helped": None,
            "reason": "GNN libraries unavailable or GNN run skipped.",
            "best_baseline": {"name": best_baseline_name, "f1": best_baseline_f1},
        }

    gnn_scores = {
        name: stats["test_metrics"]["f1"]
        for name, stats in gnn_results["models"].items()
    }
    best_gnn_name = max(gnn_scores, key=gnn_scores.get)
    best_gnn_f1 = gnn_scores[best_gnn_name]

    return {
        "graph_helped": bool(best_gnn_f1 > (best_baseline_f1 or 0.0)),
        "best_baseline": {"name": best_baseline_name, "f1": best_baseline_f1},
        "best_gnn": {"name": best_gnn_name, "f1": best_gnn_f1},
        "f1_gain": (best_gnn_f1 - best_baseline_f1) if best_baseline_f1 is not None else None,
    }


def run_experiment(data_dir: Path, config: ExperimentConfig, skip_gnn: bool = False) -> Dict[str, Any]:
    features, classes, edges = load_elliptic_data(data_dir)
    full = _prepare_feature_table(features, classes)

    report: Dict[str, Any] = {
        "data_understanding": data_understanding(features, classes, edges),
        "baselines": run_feature_baselines(full, config),
    }

    gnn_results: Optional[Dict[str, Any]] = None
    if not skip_gnn:
        try:
            gnn_results = run_gnn_models(full, edges, config)
            report["gnn_models"] = gnn_results
        except Exception as exc:
            report["gnn_models"] = {"error": f"Skipped/failed GNN execution: {exc}"}
    else:
        report["gnn_models"] = {"skipped": True}

    report["graph_structure_value_analysis"] = analyze_graph_value(report["baselines"], gnn_results)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elliptic fraud detection analysis pipeline")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing Elliptic CSV files")
    parser.add_argument("--train-end-step", type=int, default=34)
    parser.add_argument("--val-end-step", type=int, default=43)
    parser.add_argument("--gnn-epochs", type=int, default=100)
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        train_end_step=args.train_end_step,
        val_end_step=args.val_end_step,
        gnn_epochs=args.gnn_epochs,
    )

    result = run_experiment(args.data_dir, config, skip_gnn=args.skip_gnn)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
