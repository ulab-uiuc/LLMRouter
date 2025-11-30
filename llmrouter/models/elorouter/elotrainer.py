import os
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.utils import save_model


def compute_elo_mle(
    df: pd.DataFrame,
    SCALE: float = 400.0,
    BASE: float = 10.0,
    INIT_RATING: float = 1000.0,
):
    """
    Compute Elo scores via logistic regression MLE.
    """
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    idx_map = pd.Series(np.arange(len(models)), index=models)

    n = df.shape[0]
    p = len(models)

    X = np.zeros((n, p))
    X[np.arange(n), idx_map[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), idx_map[df["model_b"]]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1
    Y[df["winner"] == "model_b"] = 0

    lr = LogisticRegression(fit_intercept=False, penalty=None, solver="lbfgs")
    lr.fit(X, Y)

    elo = SCALE * lr.coef_[0] + INIT_RATING
    return pd.Series(elo, index=models).sort_values(ascending=False)


class EloRouterTrainer(BaseTrainer):
    """
    Train EloRouter by building pairwise battles and estimating Elo scores.
    """

    def __init__(self, router, optimizer=None, device="cpu"):
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.router = router
        self.routing_df = router.routing_data_train.reset_index(drop=True)

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.save_model_path = os.path.join(
            project_root, router.cfg["model_path"]["save_model_path"]
        )

        print("[EloRouterTrainer] Initialized.")

    # ---------------------------------------------------------
    # Build symmetric battles (forward + reverse)
    # ---------------------------------------------------------
    def _build_battles(self) -> pd.DataFrame:
        """
        Build balanced pairwise battles:
            winner vs loser → label = 1
            loser  vs winner → label = 0
        This ensures LogisticRegression has 2 classes.
        """
        battles = []
        grouped = self.routing_df.groupby("query")

        for q, df in grouped:
            best = df.loc[df["performance"].idxmax()]
            winner = best["model_name"]

            for _, row in df.iterrows():
                loser = row["model_name"]
                if loser == winner:
                    continue

                # Forward: winner beats loser
                battles.append({
                    "model_a": winner,
                    "model_b": loser,
                    "winner": "model_a"
                })

                # Reverse: loser loses to winner
                battles.append({
                    "model_a": loser,
                    "model_b": winner,
                    "winner": "model_b"
                })

        return pd.DataFrame(battles, columns=["model_a", "model_b", "winner"])

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    def train(self):
        battles_df = self._build_battles()

        if battles_df.empty:
            raise ValueError("[EloRouterTrainer] Empty battle table.")

        elo_scores = compute_elo_mle(battles_df)

        save_model(elo_scores, self.save_model_path)
        print(f"[EloRouterTrainer] Saved Elo scores to: {self.save_model_path}")


