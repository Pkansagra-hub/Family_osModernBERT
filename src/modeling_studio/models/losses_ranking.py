"""
Ranking Loss Functions for Multi-Granularity Relevance Head (MGRH)

Listwise and pairwise ranking losses for graded relevance training.
These losses are used in the training loop (not inside the head's forward()),
applied to the relevance scores produced by MultiGranularityRelevanceHead.

Loss Functions:
    - LambdaRankLoss: Listwise nDCG optimization via lambda gradients (Burges 2010)
    - CombinedRankingLoss: LambdaRank + pairwise margin on hard negatives

Usage:
    from modeling_studio.models.losses_ranking import (
        LambdaRankLoss,
        CombinedRankingLoss,
    )

    # Listwise ranking on human benchmark grades (0-3)
    lambda_loss = LambdaRankLoss(ndcg_at=10)
    loss = lambda_loss(scores, grades)

    # Combined: listwise + pairwise margin on hard negatives
    combined = CombinedRankingLoss(margin=0.2, alpha=0.3, ndcg_at=10)
    loss = combined(scores, grades, is_hard_negative=mask)

References:
    Burges et al., "Learning to Rank using Gradient Descent" (2006)
    Burges, "From RankNet to LambdaRank to LambdaMART" (2010)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LambdaRankLoss",
    "CombinedRankingLoss",
]


class LambdaRankLoss(nn.Module):
    """LambdaRank loss for listwise relevance training.

    Directly optimizes nDCG by weighting pairwise gradients by |delta-nDCG|.
    Requires queries with multiple documents and graded relevance labels.

    Args:
        sigma: Temperature for pairwise sigmoid. Higher values make the
            gradient sharper around the decision boundary.
        ndcg_at: Truncation depth for nDCG computation.
    """

    def __init__(self, sigma: float = 1.0, ndcg_at: int = 10) -> None:
        super().__init__()
        self.sigma = sigma
        self.ndcg_at = ndcg_at

    def forward(
        self,
        scores: torch.Tensor,
        grades: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LambdaRank loss for a single query group.

        Args:
            scores: Model scores for all docs in query group. Shape ``[N]``.
            grades: Relevance grades (0-3). Shape ``[N]``.

        Returns:
            Scalar loss tensor.
        """
        n = scores.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        gains = (2.0 ** grades.float()) - 1.0  # standard DCG gain

        # Ideal DCG for normalization
        ideal_sorted = grades.float().sort(descending=True).values[: self.ndcg_at]
        positions = torch.arange(
            2, ideal_sorted.shape[0] + 2, device=grades.device, dtype=torch.float
        )
        ideal_dcg = (((2.0 ** ideal_sorted) - 1.0) / torch.log2(positions)).sum()

        if ideal_dcg <= 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Pairwise score differences  [N, N]
        si = scores.unsqueeze(1).expand(n, n)
        sj = scores.unsqueeze(0).expand(n, n)
        diff = si - sj

        # Pairwise gain mask: i should rank above j
        gi = gains.unsqueeze(1).expand(n, n)
        gj = gains.unsqueeze(0).expand(n, n)
        relevant_pairs = (gi > gj).float()

        # |delta-nDCG| weighting
        ranks = torch.arange(1, n + 1, device=scores.device, dtype=torch.float)
        discount_i = (1.0 / torch.log2(ranks + 1)).unsqueeze(1).expand(n, n)
        discount_j = (1.0 / torch.log2(ranks + 1)).unsqueeze(0).expand(n, n)
        delta_ndcg = torch.abs(discount_i - discount_j) * torch.abs(gi - gj)
        delta_ndcg = delta_ndcg / ideal_dcg

        # Lambda gradient weighting
        lambda_ij = delta_ndcg * relevant_pairs * torch.sigmoid(-self.sigma * diff)

        # Weighted negative log-likelihood
        loss = -(lambda_ij * F.logsigmoid(self.sigma * diff)).sum()
        return loss / (n * (n - 1) + 1e-8)


class CombinedRankingLoss(nn.Module):
    """Combined LambdaRank + pairwise margin loss.

    LambdaRank provides global list-level nDCG optimization.
    Pairwise margin provides local hard-negative discrimination.

    Args:
        margin: Margin for the pairwise hinge loss between positive and
            negative scores.
        alpha: Weight for the pairwise margin component relative to
            LambdaRank (which is always weight 1.0).
        ndcg_at: Truncation depth passed to LambdaRankLoss.
    """

    def __init__(
        self,
        margin: float = 0.2,
        alpha: float = 0.3,
        ndcg_at: int = 10,
    ) -> None:
        super().__init__()
        self.lambda_loss = LambdaRankLoss(ndcg_at=ndcg_at)
        self.margin = margin
        self.alpha = alpha

    def forward(
        self,
        scores: torch.Tensor,
        grades: torch.Tensor,
        is_hard_negative: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined loss for a single query group.

        Args:
            scores: Model scores for all docs in query group. Shape ``[N]``.
            grades: Relevance grades (0-3). Shape ``[N]``.
            is_hard_negative: Boolean mask identifying hard negatives among
                the grade-0 documents. Shape ``[N]``. If ``None``, all
                grade-0 documents are treated as hard negatives.

        Returns:
            Scalar loss tensor.
        """
        ll = self.lambda_loss(scores, grades)

        # Pairwise margin on hard negatives
        pos_mask = grades > 1  # grades 2, 3 = clearly relevant
        neg_mask = grades == 0
        if is_hard_negative is not None:
            neg_mask = neg_mask & is_hard_negative

        pl = torch.tensor(0.0, device=scores.device)
        if pos_mask.any() and neg_mask.any():
            s_pos = scores[pos_mask].unsqueeze(1)  # [P, 1]
            s_neg = scores[neg_mask].unsqueeze(0)  # [1, Q]
            pl = F.relu(self.margin - (s_pos - s_neg)).mean()

        return ll + self.alpha * pl
