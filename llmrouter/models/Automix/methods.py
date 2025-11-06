"""
Automix Routing Methods
-----------------------
This module contains various routing methods for the Automix router.

Methods included:
- Threshold: Single threshold-based routing
- DoubleThreshold: Double threshold-based routing
- TripleThreshold: Triple threshold-based routing
- SelfConsistency: Self-consistency based routing
- POMDPSimple: POMDP-based routing (simple version)
- GreedyPOMDP: Greedy POMDP-based routing
- AutomixUnion: Union of multiple routing methods
- POMDP: Composite POMDP method

Original source: automix/colabs/automix_methods.py
Adapted for LLMRouter framework.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class Threshold:
    """
    Threshold-based routing method.

    Routes to large model when verifier score <= threshold.
    """

    def __init__(self, num_bins: int, **kwargs):
        """
        Args:
            num_bins: Number of bins for discretizing verification scores
            **kwargs: Additional keyword arguments stored as attributes
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.num_bins = num_bins
        self.gap = 1 / num_bins

    def run(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5,
        verifier_column: str = "p_ver_13b",
    ) -> pd.Series:
        """
        Determine routing decisions based on threshold.

        Args:
            data: Input dataframe
            threshold: Threshold value for routing
            verifier_column: Column name containing verification scores

        Returns:
            Boolean series indicating whether to route to large model
        """
        to_retry = data[verifier_column] <= threshold
        return to_retry

    def generate_points(
        self, data: pd.DataFrame = None, verifier_column: str = "p_ver_13b"
    ) -> List[float]:
        """
        Generate candidate threshold points for training.

        Args:
            data: Input dataframe (not used for simple threshold)
            verifier_column: Column name containing verification scores

        Returns:
            List of candidate threshold values
        """
        gap = 1 / self.num_bins
        return [x * gap for x in range(self.num_bins + 1)]


class DoubleThreshold(Threshold):
    """
    Double threshold-based routing method.

    Routes to large model when verifier score is between two thresholds.
    """

    def run(
        self,
        data: pd.DataFrame,
        threshold: Tuple[float, float] = (0.25, 0.75),
        verifier_column: str = "p_ver_13b",
    ) -> pd.Series:
        """
        Determine routing decisions based on double threshold.

        Args:
            data: Input dataframe
            threshold: Tuple of (lower, upper) thresholds
            verifier_column: Column name containing verification scores

        Returns:
            Boolean series indicating whether to route to large model
        """
        try:
            to_retry = data[verifier_column].between(threshold[0], threshold[1])
        except:
            to_retry = data[[verifier_column]].between(threshold[0], threshold[1])[
                verifier_column
            ]
        return to_retry

    def generate_points(
        self, data: pd.DataFrame = None, verifier_column: str = "p_ver_13b"
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate double threshold points for training.

        Args:
            data: Input dataframe (not used)
            verifier_column: Column name containing verification scores

        Returns:
            List of candidate (lower, upper) threshold tuples
        """
        points = []
        for i in range(self.num_bins):
            for j in range(i + 1, self.num_bins):
                points.append((i * self.gap, j * self.gap))
        return points


class TripleThreshold(DoubleThreshold):
    """
    Triple threshold-based routing method.

    Routes to large model when verifier score is in certain ranges.
    """

    def run(
        self,
        data: pd.DataFrame,
        threshold: Tuple[float, float, float] = (0.25, 0.5, 0.75),
        verifier_column: str = "p_ver_13b",
    ) -> pd.Series:
        """
        Determine routing decisions based on triple threshold.

        Args:
            data: Input dataframe
            threshold: Tuple of three threshold values
            verifier_column: Column name containing verification scores

        Returns:
            Boolean series indicating whether to route to large model
        """
        try:
            to_retry = data[verifier_column].between(0, threshold[0]) | data[
                verifier_column
            ].between(threshold[1], threshold[2])
        except:
            to_retry = (
                data[[verifier_column]].between(0, threshold[0])[verifier_column]
                | data[[verifier_column]].between(threshold[1], threshold[2])[
                    verifier_column
                ]
            )
        return to_retry

    def generate_points(
        self, data: pd.DataFrame = None, verifier_column: str = "p_ver_13b"
    ) -> List[Tuple[float, float, float]]:
        """
        Generate candidate triple threshold points for training.

        Args:
            data: Input dataframe (not used)
            verifier_column: Column name containing verification scores

        Returns:
            List of candidate three-threshold tuples
        """
        points = []
        for i in range(self.num_bins):
            for j in range(i + 1, self.num_bins):
                for k in range(j + 1, self.num_bins):
                    points.append((i * self.gap, j * self.gap, k * self.gap))
        return points


class SelfConsistency(Threshold):
    """
    Self-consistency based routing method.

    Uses a fixed threshold of 0.5 for routing decisions.
    """

    def generate_points(
        self, data: pd.DataFrame = None, verifier_column: str = "p_ver_13b"
    ) -> List[float]:
        """
        Generate candidate points (fixed at 0.5 for self-consistency).

        Args:
            data: Input dataframe (not used)
            verifier_column: Column name containing verification scores

        Returns:
            List containing single threshold value of 0.5
        """
        return [0.5]


class POMDPSimple:
    """
    POMDP-based routing method (simple version).

    Uses Partially Observable Markov Decision Process for routing decisions.
    """

    def __init__(self, num_bins: int, init_belief: bool = False, **kwargs):
        """
        Args:
            num_bins: Number of bins for discretizing verification scores
            init_belief: Whether to use initial belief from data
            **kwargs: Additional keyword arguments
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.num_bins = num_bins
        self.gap = 1 / num_bins
        self.init_belief = init_belief

    def compute_obs_probs(
        self, df: pd.DataFrame, verifier_column: str = "p_ver_13b"
    ) -> List[Tuple[int]]:
        """
        Compute observation probabilities and generate action sequences.

        Args:
            df: Input dataframe
            verifier_column: Column name containing verification scores

        Returns:
            List of unique action sequences
        """
        categories = ["NEEDY", "GOOD", "HOPELESS"]
        obs_probs = np.zeros((self.num_bins + 1, 3))

        for idx, prob in enumerate([i * self.gap for i in range(self.num_bins + 1)]):
            df_new = df[df[verifier_column] - prob < self.gap / 2]
            df_new = df_new[df_new[verifier_column] - prob > -self.gap / 2]
            try:
                vcs = df_new["category"].value_counts()
                obs_probs[idx] = [
                    (vcs[cat] if cat in vcs else 0) / len(df_new) for cat in categories
                ]
            except Exception:
                pass

        belief = np.array([1 for _ in range(len(categories))])
        if self.init_belief:
            belief = np.array(
                [
                    (
                        df["category"].value_counts()[cat]
                        if cat in df["category"].value_counts()
                        else 0
                    )
                    for cat in categories
                ]
            )

        belief = belief / sum(belief)

        action_seqs = []

        for reward in range(5000):
            action_array = np.array(
                [[-reward, 0, -reward - 1], [-100, -100, -reward - 100]]
            )
            actions = []
            for i in range(self.num_bins + 1):
                scores = obs_probs[i] * action_array
                actions.append(np.argmax(scores.sum(axis=1)))
            action_seqs.append(tuple(actions))

        return list(set(action_seqs))

    def get_nearest_prob_idx(self, prob: float) -> int:
        """
        Get nearest bin index for a probability value.

        Args:
            prob: Probability value

        Returns:
            Bin index
        """
        for i in [self.gap * i for i in range(self.num_bins + 1)]:
            if prob - i < self.gap / 2 and prob - i >= -self.gap / 2:
                return int(i // self.gap)
        return 0

    def get_action(self, x: float, action_seq: Tuple[int]) -> bool:
        """
        Get action for a given verification score.

        Args:
            x: Verification score
            action_seq: Action sequence

        Returns:
            Boolean indicating routing decision
        """
        return action_seq[self.get_nearest_prob_idx(x)] == 1

    def run(
        self,
        data: pd.DataFrame,
        action_seq: List[int] = [],
        verifier_column: str = "p_ver_13b",
    ) -> pd.Series:
        """
        Determine routing decisions based on POMDP action sequence.

        Args:
            data: Input dataframe
            action_seq: Action sequence from training
            verifier_column: Column name containing verification scores

        Returns:
            Boolean series indicating whether to route to large model
        """
        value = data[verifier_column]

        try:
            is_scalar = pd.api.types.is_scalar
        except Exception:
            from numpy import isscalar as is_scalar

        if is_scalar(value):
            return self.get_action(value, action_seq)
        else:
            return value.apply(lambda x: self.get_action(x, action_seq))

    def generate_points(
        self, data: pd.DataFrame, verifier_column: str = "p_ver_13b"
    ) -> List[Tuple[int]]:
        """
        Generate candidate action sequences for training.

        Args:
            data: Input dataframe
            verifier_column: Column name containing verification scores

        Returns:
            List of action sequences
        """
        return self.compute_obs_probs(data, verifier_column=verifier_column)


class GreedyPOMDP(POMDPSimple):
    """
    Greedy POMDP-based routing method.

    Uses greedy approach to select routing actions based on performance gain.
    """

    def __init__(
        self,
        num_bins: int,
        init_belief: bool = False,
        slm_column: str = "llama13b_f1",
        llm_column: str = "llama70b_f1",
        **kwargs
    ):
        """
        Args:
            num_bins: Number of bins for discretizing verification scores
            init_belief: Whether to use initial belief from data
            slm_column: Column name for small model scores
            llm_column: Column name for large model scores
            **kwargs: Additional keyword arguments
        """
        super().__init__(num_bins, init_belief, **kwargs)
        self.slm_column = slm_column
        self.llm_column = llm_column

    def generate_points(
        self, data: pd.DataFrame = None, verifier_column: str = "p_ver_13b"
    ) -> List[List[int]]:
        """
        Generate candidate action sequences using greedy approach.

        Args:
            data: Input dataframe
            verifier_column: Column name containing verification scores

        Returns:
            List of action sequences (as lists of 0/1)
        """
        points = []
        for x in [i for i in range(self.num_bins + 1)]:
            df_fil = data[
                data[verifier_column].apply(lambda y: self.get_nearest_prob_idx(y) == x)
            ]
            df_fil = data[data[verifier_column].apply(lambda y: y // self.gap == x)]

            wt = len(df_fil) / len(data)
            delta_f = df_fil[self.llm_column] - df_fil[self.slm_column]
            points.append((delta_f.mean() if len(df_fil) != 0 else 0, x, wt))

        points = list(
            zip(
                gaussian_filter([x[0] for x in points], sigma=2),
                [x[1] for x in points],
                [x[2] for x in points],
            )
        )

        params = [
            [0 for _ in range(self.num_bins + 1)] for _ in range(len(points))
        ]
        total_perf = 0
        total_cost = 1

        for i, index in enumerate(sorted(points[:], key=lambda x: x[0])[::-1]):
            if i == 0:
                params[i][index[1]] = 1
            else:
                idx = index[1]
                params[i] = params[i - 1].copy()
                params[i][idx] = 1
            total_perf += index[0] * index[2]
            total_cost += index[2] * 49

        return params


class AutomixUnion:
    """
    Union of multiple routing methods.

    Allows combining different routing strategies.
    """

    def __init__(self, *methods):
        """
        Args:
            *methods: Variable number of routing method instances
        """
        self.methods = list(methods)

    def run(
        self,
        data: pd.DataFrame,
        param: Tuple,
        verifier_column: str = "p_ver_13b",
    ) -> pd.Series:
        """
        Run routing using one of the combined methods.

        Args:
            data: Input dataframe
            param: Tuple of (method_param, method_index)
            verifier_column: Column name containing verification scores

        Returns:
            Boolean series indicating routing decisions
        """
        return self.methods[param[1]].run(
            data, param[0], verifier_column=verifier_column
        )

    def generate_points(
        self, data: pd.DataFrame = None, verifier_column: str = "p_ver_13b"
    ) -> List[Tuple]:
        """
        Generate candidate points from all combined methods.

        Args:
            data: Input dataframe
            verifier_column: Column name containing verification scores

        Returns:
            List of (method_param, method_index) tuples
        """
        new_points = []
        for i, meth in enumerate(self.methods):
            new_points.extend(
                [(x, i) for x in meth.generate_points(data, verifier_column=verifier_column)]
            )
        return new_points

    def __repr__(self) -> str:
        return "AutomixUnion(" + ", ".join([str(x) for x in self.methods]) + ")"


class FixedAnswerRouting:
    """
    Routing method that combines fixed answer routing with another method.

    Routes to large model for specific fixed answers OR based on another method.
    """

    def __init__(
        self,
        method,
        fixed_routing_elems: List[str] = [],
        ans_column: str = "llama13b_pred_ans",
    ):
        """
        Args:
            method: Base routing method
            fixed_routing_elems: List of answers that always route to large model
            ans_column: Column name containing answers
        """
        self.fixed_routing_elems = fixed_routing_elems
        self.method = method
        self.ans_column = ans_column

    def run(
        self,
        data: pd.DataFrame,
        param,
        verifier_column: str = "p_ver_13b",
    ) -> pd.Series:
        """
        Determine routing decisions combining fixed and method-based routing.

        Args:
            data: Input dataframe
            param: Parameter for base method
            verifier_column: Column name containing verification scores

        Returns:
            Boolean series indicating routing decisions
        """
        if isinstance(data[self.ans_column], str):
            to_retry = data[[self.ans_column]].apply(
                lambda x: x in self.fixed_routing_elems
            )[self.ans_column]
        else:
            to_retry = data[self.ans_column].apply(
                lambda x: x in self.fixed_routing_elems
            )
        to_retry = to_retry | self.method.run(data, param, verifier_column=verifier_column)
        return to_retry

    def generate_points(
        self, data: pd.DataFrame = None, verifier_column: str = "p_ver_13b"
    ) -> List:
        """
        Generate candidate points from base method.

        Args:
            data: Input dataframe
            verifier_column: Column name containing verification scores

        Returns:
            List of candidate points
        """
        return self.method.generate_points(data, verifier_column=verifier_column)

    def __repr__(self) -> str:
        return (
            "FixedAnswerRouting("
            + str(self.method)
            + ", "
            + str(self.fixed_routing_elems)
            + ")"
        )


# POMDP is a composite method combining multiple strategies
def POMDP(*args, **kwargs):
    """
    Create a composite POMDP routing method.

    Combines POMDPSimple, GreedyPOMDP, DoubleThreshold, and Threshold methods.

    Args:
        *args: Positional arguments passed to all methods
        **kwargs: Keyword arguments passed to all methods

    Returns:
        AutomixUnion instance combining multiple methods
    """
    return AutomixUnion(
        POMDPSimple(*args, **kwargs),
        GreedyPOMDP(*args, **kwargs),
        DoubleThreshold(*args, **kwargs),
        Threshold(*args, **kwargs),
    )
