"""Candidate classifiers used to rank road segments for GPS observations."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return five distinct candidate-ranking models with consistent imputation."""
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08, l2_regularization=1.0, random_state=random_state)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.06, max_depth=2, random_state=random_state)),
        ]),
    }


def positive_probabilities(estimator: Any, features: Any) -> Any:
    """Return class-1 probabilities, including estimators with one class only."""
    classes = estimator.named_steps["classifier"].classes_
    probabilities = estimator.predict_proba(features)
    if 1 not in classes:
        return probabilities[:, 0] * 0.0
    return probabilities[:, list(classes).index(1)]


try:
    import torch
    from torch import nn
    from torch_geometric.nn import SAGEConv

    class RoadGraphSAGE(nn.Module):
        """Optional PyG node scorer for road arcs on an arc-adjacency graph.

        Each node represents one road arc; ``edge_index`` links arcs whose
        endpoint nodes touch. Candidate and GPS-derived numeric features are
        supplied per arc as ``x``. This model is an experiment scaffold and is
        intentionally not selected automatically by the tabular CV runner.
        """

        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
            """Initialize two graph-convolution layers and an arc scoring head."""
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Return a logit for each candidate arc node."""
            hidden = self.dropout(torch.relu(self.conv1(x, edge_index)))
            hidden = self.dropout(torch.relu(self.conv2(hidden, edge_index)))
            return self.output(hidden).squeeze(-1)

except ImportError:
    RoadGraphSAGE = None  # type: ignore[assignment,misc]
