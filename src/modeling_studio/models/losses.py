"""
Loss Functions for Multi-Task Training

This module contains specialized loss functions for various tasks
in the multi-task learning setup.

Loss Functions:
    - FocalLoss: For class-imbalanced classification (safety bands)
    - LabelSmoothingCrossEntropy: Regularized classification
    - MultipleNegativesRankingLoss: Contrastive learning for embeddings
    - CosineSimilarityLoss: Regression on similarity scores
    - TripletLoss: Anchor-positive-negative embedding learning
    - CRFLoss: Conditional Random Field for sequence labeling
    - FamilyContrastiveLoss: Family-aware contrastive learning with hard negatives

Multi-Task Losses:
    - MultiTaskLoss: Weighted combination of task losses
    - UncertaintyWeightedLoss: Learn task weights automatically (Kendall et al.)

Usage:
    from modeling_studio.models.losses import FocalLoss, UncertaintyWeightedLoss

    # Focal loss for imbalanced safety classification
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(logits, targets)

    # Uncertainty weighting for multi-task learning
    uw_loss = UncertaintyWeightedLoss(num_tasks=5)
    combined_loss = uw_loss(task_losses=[loss1, loss2, loss3, loss4, loss5])

    # Family contrastive loss for embeddings
    family_loss = FamilyContrastiveLoss(temperature=0.07)
    loss = family_loss(anchor, positive, negatives)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# FocalLoss - Class Imbalance Handling
# =============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification tasks.

    Focal loss down-weights easy examples and focuses training on hard negatives.
    Particularly useful for safety classification where CRISIS class is rare.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        - p_t is the probability of the correct class
        - gamma is the focusing parameter (higher = more focus on hard examples)
        - alpha is the class weight (can be scalar or per-class tensor)

    Args:
        alpha: Class weight(s). Can be:
            - float: Applied to all classes
            - list/tensor: Per-class weights
            - None: No class weighting
        gamma: Focusing parameter. Default: 2.0
            - gamma=0: Equivalent to cross entropy
            - gamma>0: Down-weights easy examples
        reduction: Loss reduction method ('mean', 'sum', 'none')
        ignore_index: Label index to ignore (e.g., padding). Default: -100

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    Example:
        >>> focal = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 4)  # 4 safety bands
        >>> labels = torch.randint(0, 4, (32,))
        >>> loss = focal(logits, labels)
        >>> assert loss.requires_grad
    """

    def __init__(
        self,
        alpha: float | list[float] | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        # Handle alpha (class weights)
        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor([alpha]))
        elif isinstance(alpha, list):
            self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Focal loss value
        """
        num_classes = logits.size(-1)

        # Compute cross entropy (without reduction)
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            targets.view(-1),
            reduction="none",
            ignore_index=self.ignore_index,
        )

        # Compute probability of target class
        pt = torch.exp(-ce_loss)

        # Apply focal modulation
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if alpha.numel() == 1:
                # Scalar alpha - apply uniformly
                alpha_t = alpha.item()
            else:
                # Per-class alpha - gather by target
                valid_mask = targets.view(-1) != self.ignore_index
                alpha_t = torch.ones_like(ce_loss)
                alpha_t[valid_mask] = alpha[targets.view(-1)[valid_mask]]
            focal_weight = alpha_t * focal_weight

        # Apply focal loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# =============================================================================
# LabelSmoothingCrossEntropy - Regularization
# =============================================================================


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing for regularization.

    Label smoothing prevents overconfident predictions by replacing hard
    one-hot targets with soft targets that reserve some probability mass
    for incorrect classes.

    Formula:
        Smoothed target = (1 - epsilon) * one_hot + epsilon / num_classes

    Args:
        epsilon: Smoothing factor (0 = no smoothing, 1 = uniform distribution)
            Default: 0.1
        reduction: Loss reduction method ('mean', 'sum', 'none')
        ignore_index: Label index to ignore. Default: -100

    Reference:
        Szegedy et al. "Rethinking the Inception Architecture" (CVPR 2016)

    Example:
        >>> loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
        >>> logits = torch.randn(32, 10)
        >>> labels = torch.randint(0, 10, (32,))
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy loss.

        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Label-smoothed loss value
        """
        num_classes = logits.size(-1)

        # Create mask for valid (non-ignored) positions
        valid_mask = targets != self.ignore_index

        # Log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute smoothed loss
        # Loss = (1 - epsilon) * ce_loss + epsilon * uniform_loss
        # uniform_loss = -sum(log_probs) / num_classes = -mean(log_probs)

        # Standard cross entropy part
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1).clamp(min=0)).squeeze(-1)

        # Uniform distribution part (sum of all log probs)
        smooth_loss = -log_probs.sum(dim=-1) / num_classes

        # Combine with smoothing
        loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss

        # Apply mask
        loss = loss * valid_mask.float()

        # Apply reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# =============================================================================
# MultipleNegativesRankingLoss - Contrastive Embedding Loss
# =============================================================================


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss for contrastive embedding learning.

    Uses in-batch negatives: for each (anchor, positive) pair, all other
    positives in the batch serve as negatives. This is efficient as it
    doesn't require explicit negative sampling.

    Formula:
        For anchor a and positive p:
        L = -log(exp(sim(a, p) / τ) / Σ_j exp(sim(a, p_j) / τ))

    Where τ (temperature) controls the sharpness of the distribution.

    Args:
        scale: Temperature scale (1/τ). Higher = sharper distribution. Default: 20.0
        similarity_fn: Similarity function. Default: cosine similarity

    Reference:
        Henderson et al. "Efficient Natural Language Response Suggestion" (ACL 2017)
        Sentence-BERT uses this for training embeddings

    Example:
        >>> loss_fn = MultipleNegativesRankingLoss(scale=20.0)
        >>> embeddings_a = torch.randn(32, 768)  # Anchors
        >>> embeddings_b = torch.randn(32, 768)  # Positives
        >>> loss = loss_fn(embeddings_a, embeddings_b)
    """

    def __init__(
        self,
        scale: float = 20.0,
        similarity_fn: str = "cosine",
    ):
        super().__init__()
        self.scale = scale
        self.similarity_fn = similarity_fn

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute multiple negatives ranking loss.

        Args:
            embeddings_a: Anchor embeddings (batch_size, embedding_dim)
            embeddings_b: Positive embeddings (batch_size, embedding_dim)
            labels: Optional explicit labels. If None, assumes diagonal is positive.

        Returns:
            Contrastive loss value
        """
        # Normalize embeddings for cosine similarity
        if self.similarity_fn == "cosine":
            embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
            embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Compute similarity matrix (batch_size x batch_size)
        similarity_matrix = torch.matmul(embeddings_a, embeddings_b.T) * self.scale

        # Labels: diagonal elements are positives (identity matrix indices)
        batch_size = embeddings_a.size(0)
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings_a.device)

        # Cross entropy loss treats this as multi-class classification
        # Each anchor should match its corresponding positive
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


# =============================================================================
# CosineSimilarityLoss - Embedding Similarity Training
# =============================================================================


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for embedding similarity regression.

    Trains embeddings such that their cosine similarity matches target scores.
    Used for tasks like Semantic Textual Similarity (STS).

    Formula:
        L = MSE(cosine_sim(a, b), target_similarity)

    Args:
        loss_fn: Underlying loss function ('mse', 'smooth_l1'). Default: 'mse'

    Example:
        >>> loss_fn = CosineSimilarityLoss()
        >>> embeddings_a = torch.randn(32, 768)
        >>> embeddings_b = torch.randn(32, 768)
        >>> targets = torch.rand(32)  # Similarity scores [0, 1]
        >>> loss = loss_fn(embeddings_a, embeddings_b, targets)
    """

    def __init__(self, loss_fn: Literal["mse", "smooth_l1"] = "mse"):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.

        Args:
            embeddings_a: First set of embeddings (batch_size, embedding_dim)
            embeddings_b: Second set of embeddings (batch_size, embedding_dim)
            targets: Target similarity scores (batch_size,) in [0, 1] or [-1, 1]

        Returns:
            Similarity regression loss
        """
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embeddings_a, embeddings_b, dim=-1)

        # Compute loss
        if self.loss_fn == "mse":
            loss = F.mse_loss(cosine_sim, targets)
        else:  # smooth_l1
            loss = F.smooth_l1_loss(cosine_sim, targets)

        return loss


# =============================================================================
# TripletLoss - Triplet Margin Loss for Embeddings
# =============================================================================


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss for embedding learning with hard negative mining.

    Trains embeddings such that anchors are closer to positives than negatives
    by at least a margin.

    Formula:
        L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Supports hard negative mining to focus on difficult examples.

    Args:
        margin: Minimum margin between positive and negative distances. Default: 1.0
        distance_fn: Distance function ('euclidean', 'cosine'). Default: 'euclidean'
        hard_negative_mining: Whether to mine hard negatives. Default: False
        swap: Whether to use distance swap (FaceNet paper). Default: False

    Reference:
        Schroff et al. "FaceNet: A Unified Embedding" (CVPR 2015)

    Example:
        >>> loss_fn = TripletLoss(margin=0.5, distance_fn='cosine')
        >>> anchor = torch.randn(32, 768)
        >>> positive = torch.randn(32, 768)
        >>> negative = torch.randn(32, 768)
        >>> loss = loss_fn(anchor, positive, negative)
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_fn: Literal["euclidean", "cosine"] = "euclidean",
        hard_negative_mining: bool = False,
        swap: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.hard_negative_mining = hard_negative_mining
        self.swap = swap

        # Use PyTorch's built-in triplet loss for standard case
        if distance_fn == "euclidean" and not hard_negative_mining:
            self._triplet_loss = nn.TripletMarginLoss(
                margin=margin,
                p=2,
                swap=swap,
                reduction="mean",
            )
        else:
            self._triplet_loss = None

    def _compute_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance."""
        if self.distance_fn == "euclidean":
            return torch.pairwise_distance(a, b, p=2)
        else:  # cosine
            # Cosine distance = 1 - cosine_similarity
            return 1 - F.cosine_similarity(a, b, dim=-1)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)

        Returns:
            Triplet margin loss
        """
        if self._triplet_loss is not None:
            return self._triplet_loss(anchor, positive, negative)

        # Custom implementation for cosine or hard mining
        d_ap = self._compute_distance(anchor, positive)
        d_an = self._compute_distance(anchor, negative)

        if self.swap:
            d_pn = self._compute_distance(positive, negative)
            d_an = torch.min(d_an, d_pn)

        if self.hard_negative_mining:
            # Semi-hard negative mining: select negatives where
            # d(a, p) < d(a, n) < d(a, p) + margin
            mask = (d_an > d_ap) & (d_an < d_ap + self.margin)
            if mask.sum() > 0:
                losses = F.relu(d_ap[mask] - d_an[mask] + self.margin)
                return losses.mean()
            # Fallback to all triplets
            losses = F.relu(d_ap - d_an + self.margin)
            return losses.mean()

        # Standard triplet loss
        losses = F.relu(d_ap - d_an + self.margin)
        return losses.mean()


# =============================================================================
# CRFLoss - Conditional Random Field for NER
# =============================================================================


class CRFLoss(nn.Module):
    """
    Conditional Random Field (CRF) layer for sequence labeling (NER).

    Models label dependencies using transition scores between consecutive
    labels, improving entity boundary detection.

    Components:
        - Emission scores: From encoder (logits)
        - Transition scores: Learned (A[i,j] = score for j -> i transition)

    Training:
        L = -log P(y | x) = -S(x, y) + log Σ_y' exp(S(x, y'))

    Inference:
        Uses Viterbi decoding for optimal label sequence.

    Args:
        num_tags: Number of NER tags
        batch_first: Whether batch dimension is first. Default: True
        pad_tag_id: ID of padding tag. Default: None (no constraint)

    Reference:
        Lafferty et al. "Conditional Random Fields" (ICML 2001)
        Lample et al. "Neural Architectures for NER" (NAACL 2016)

    Example:
        >>> crf = CRFLoss(num_tags=17)
        >>> emissions = torch.randn(32, 128, 17)  # (batch, seq, tags)
        >>> tags = torch.randint(0, 17, (32, 128))
        >>> mask = torch.ones(32, 128, dtype=torch.bool)
        >>> loss = crf(emissions, tags, mask)
    """

    def __init__(
        self,
        num_tags: int,
        batch_first: bool = True,
        pad_tag_id: int | None = None,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.pad_tag_id = pad_tag_id

        # Transition parameters
        # transitions[i, j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Start and end transition scores
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_transitions()

    def _init_transitions(self) -> None:
        """Initialize transition scores."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        # Constraints: cannot transition from/to padding if specified
        if self.pad_tag_id is not None:
            with torch.no_grad():
                self.transitions[self.pad_tag_id, :] = -10000.0
                self.transitions[:, self.pad_tag_id] = -10000.0
                self.start_transitions[self.pad_tag_id] = -10000.0
                self.end_transitions[self.pad_tag_id] = -10000.0

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            emissions: Emission scores (batch, seq_len, num_tags)
            tags: Ground truth tags (batch, seq_len)
            mask: Sequence mask (batch, seq_len). True = valid.
            reduction: Loss reduction method.

        Returns:
            CRF loss
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        seq_len, batch_size, num_tags = emissions.shape

        if mask is None:
            mask = torch.ones(seq_len, batch_size, dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        # Compute score of gold sequence
        gold_score = self._score_sequence(emissions, tags, mask)

        # Compute log partition function (normalization)
        log_partition = self._compute_log_partition(emissions, mask)

        # Negative log-likelihood
        nll = log_partition - gold_score

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    def _score_sequence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score the gold tag sequence."""
        seq_len, batch_size, _ = emissions.shape

        # Start transition + first emission
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for t in range(1, seq_len):
            # Emission score
            emit_score = emissions[t, torch.arange(batch_size), tags[t]]
            # Transition score
            trans_score = self.transitions[tags[t], tags[t - 1]]

            # Add scores where mask is valid
            score += (emit_score + trans_score) * mask[t].float()

        # End transition (at last valid position)
        # Find last valid position for each sequence
        seq_lengths = mask.sum(dim=0).long()
        last_tags = tags.gather(0, (seq_lengths - 1).unsqueeze(0)).squeeze(0)
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log partition function using forward algorithm."""
        seq_len, batch_size, num_tags = emissions.shape

        # Initialize: start_transitions + first emission
        # alpha[b, t] = log Σ_paths score(path ending in tag t at position 0)
        alpha = self.start_transitions.unsqueeze(0) + emissions[0]  # (batch, num_tags)

        for t in range(1, seq_len):
            # Broadcast for all possible transitions
            # alpha_expanded: (batch, num_tags, 1)
            # transitions: (num_tags, num_tags) -> (1, num_tags, num_tags)
            # emissions: (batch, num_tags) -> (batch, 1, num_tags)
            alpha_expanded = alpha.unsqueeze(2)  # (batch, num_tags, 1)
            trans_expanded = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            emit_expanded = emissions[t].unsqueeze(1)  # (batch, 1, num_tags)

            # Score for all transitions: (batch, num_tags, num_tags)
            scores = alpha_expanded + trans_expanded + emit_expanded

            # Log-sum-exp over previous tags
            new_alpha = torch.logsumexp(scores, dim=1)  # (batch, num_tags)

            # Apply mask: keep old alpha where mask is False
            alpha = torch.where(mask[t].unsqueeze(1), new_alpha, alpha)

        # Add end transitions and compute final partition
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """
        Viterbi decoding to find optimal tag sequence.

        Args:
            emissions: Emission scores (batch, seq_len, num_tags)
            mask: Sequence mask (batch, seq_len)

        Returns:
            List of tag sequences for each batch element
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        seq_len, batch_size, num_tags = emissions.shape

        if mask is None:
            mask = torch.ones(seq_len, batch_size, dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        # Viterbi algorithm
        # delta[b, t] = best score ending in tag t
        # backpointers[t, b, tag] = best previous tag
        delta = self.start_transitions.unsqueeze(0) + emissions[0]
        backpointers: list[torch.Tensor] = []

        for t in range(1, seq_len):
            delta_expanded = delta.unsqueeze(2)  # (batch, num_tags, 1)
            trans_expanded = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)

            scores = delta_expanded + trans_expanded  # (batch, num_tags, num_tags)
            max_scores, best_tags = scores.max(dim=1)  # (batch, num_tags)

            delta = max_scores + emissions[t]
            backpointers.append(best_tags)

        # Add end transitions
        delta = delta + self.end_transitions.unsqueeze(0)

        # Backtrack
        best_paths: list[list[int]] = []
        seq_lengths = mask.sum(dim=0).long()

        for b in range(batch_size):
            seq_len_b = seq_lengths[b].item()
            best_last_tag = delta[b].argmax().item()
            best_path = [best_last_tag]

            for t in range(seq_len_b - 2, -1, -1):
                best_last_tag = backpointers[t][b, best_last_tag].item()
                best_path.insert(0, best_last_tag)

            best_paths.append(best_path)

        return best_paths


# =============================================================================
# MultiTaskLoss - Static Weight Combination
# =============================================================================


class MultiTaskLoss(nn.Module):
    """
    Combine losses from multiple tasks with static weights.

    Simple weighted sum of task losses for multi-task learning.

    Args:
        task_weights: Dictionary mapping task name to weight.
            Missing tasks default to weight 1.0.
        loss_scale: Global scale factor for stability. Default: 1.0

    Example:
        >>> mtl = MultiTaskLoss(task_weights={'safety': 15.0, 'ner': 1.0, 'sentiment': 1.0})
        >>> losses = {'safety': loss1, 'ner': loss2, 'sentiment': loss3}
        >>> total_loss = mtl(losses)
    """

    def __init__(
        self,
        task_weights: dict[str, float] | None = None,
        loss_scale: float = 1.0,
    ):
        super().__init__()
        self.task_weights = task_weights or {}
        self.loss_scale = loss_scale

    def forward(self, task_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of task losses.

        Args:
            task_losses: Dictionary mapping task name to loss tensor.

        Returns:
            Weighted total loss.
        """
        total_loss = torch.tensor(0.0, device=next(iter(task_losses.values())).device)

        for task_name, loss in task_losses.items():
            if loss is None or (isinstance(loss, torch.Tensor) and loss.numel() == 0):
                continue

            weight = self.task_weights.get(task_name, 1.0)
            total_loss = total_loss + weight * loss

        return total_loss * self.loss_scale

    def get_weights(self) -> dict[str, float]:
        """Return current task weights."""
        return self.task_weights.copy()


# =============================================================================
# UncertaintyWeightedLoss - Kendall Uncertainty Weighting (CRITICAL)
# =============================================================================


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty Weighted Multi-Task Loss (Kendall et al., 2018).

    CRITICAL component for multi-task learning. Learns task weights automatically
    based on homoscedastic uncertainty, balancing tasks without manual tuning.

    For each task i, learns log(σ_i²) where σ_i is the task uncertainty.
    The weighted loss becomes:
        L_i_weighted = (1 / (2 * σ_i²)) * L_i + log(σ_i)
                     = 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i

    This allows:
    - High-variance (noisy) tasks to have lower weight
    - Low-variance (confident) tasks to have higher weight
    - Automatic balancing without manual weight tuning

    Args:
        num_tasks: Number of tasks to weight.
        task_names: Optional list of task names for logging.
        init_value: Initial log variance value. Default: 0.0 (σ=1)

    Reference:
        Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty to
        Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)

    Example:
        >>> uw_loss = UncertaintyWeightedLoss(num_tasks=5)
        >>> task_losses = [loss_ner, loss_sent, loss_safety, loss_nli, loss_embed]
        >>> combined_loss = uw_loss(task_losses)
        >>> assert combined_loss.requires_grad
        >>> print("Task weights:", uw_loss.get_weights())
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: list[str] | None = None,
        init_value: float = 0.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]

        # Learnable log variances (one per task)
        # log(σ²) parameterization is more stable than σ directly
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_value))

    def forward(
        self,
        task_losses: list[torch.Tensor] | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute uncertainty-weighted sum of losses.

        Args:
            task_losses: Either:
                - List of task losses (scalars) in order
                - Dictionary mapping task name to loss

        Returns:
            Weighted total loss (with uncertainty regularization).
        """
        # Convert dict to list if needed
        if isinstance(task_losses, dict):
            losses = [task_losses.get(name) for name in self.task_names]
        else:
            losses = task_losses

        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")

        # Get device from first valid loss
        device = None
        for loss in losses:
            if loss is not None and isinstance(loss, torch.Tensor):
                device = loss.device
                break

        if device is None:
            raise ValueError("No valid losses provided")

        total_loss = torch.tensor(0.0, device=device)

        for i, loss in enumerate(losses):
            if loss is None or (isinstance(loss, torch.Tensor) and loss.numel() == 0):
                continue

            # Ensure loss is on correct device
            if loss.device != device:
                loss = loss.to(device)

            # precision = 1 / σ² = exp(-log(σ²))
            precision = torch.exp(-self.log_vars[i])

            # Weighted loss + regularization
            # L_weighted = (1 / (2σ²)) * L + log(σ) = 0.5 * precision * L + 0.5 * log_var
            total_loss = total_loss + 0.5 * precision * loss + 0.5 * self.log_vars[i]

        return total_loss

    def get_weights(self) -> dict[str, float]:
        """Get current task weights (1 / σ² = exp(-log_var))."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars).cpu().tolist()
        return {name: w for name, w in zip(self.task_names, weights)}

    def get_log_vars(self) -> dict[str, float]:
        """Get current log variance values."""
        with torch.no_grad():
            log_vars = self.log_vars.cpu().tolist()
        return {name: lv for name, lv in zip(self.task_names, log_vars)}

    def get_uncertainties(self) -> dict[str, float]:
        """Get current uncertainty values (σ)."""
        with torch.no_grad():
            sigmas = torch.exp(0.5 * self.log_vars).cpu().tolist()
        return {name: s for name, s in zip(self.task_names, sigmas)}


# =============================================================================
# FamilyContrastiveLoss - Family-Aware Contrastive Learning
# =============================================================================


class FamilyContrastiveLoss(nn.Module):
    """
    Family-aware contrastive loss for learning embeddings that capture
    family relationships and temporal context.

    This loss function combines InfoNCE with hard negative mining strategies
    specific to family relationships:
        - Same person in different events (photo albums)
        - Temporal neighbors (consecutive messages/events)
        - Family member disambiguation (distinguish aunts from mothers)

    The loss uses temperature-scaled cosine similarity with optional
    hard negative mining to improve embedding quality.

    Features:
        - Temperature-scaled InfoNCE loss
        - Hard negative mining support
        - In-batch negative sampling
        - Configurable negative weighting
        - Support for pre-computed hard negatives

    Args:
        temperature: Temperature scaling factor (default 0.07, lower = sharper)
        reduction: How to reduce batch losses ('mean', 'sum', 'none')
        hard_negative_weight: Extra weight for hard negatives (default 1.0 = no extra)
        use_hard_negatives: Whether to use hard negative mining
        normalize: Whether to L2-normalize embeddings

    Example:
        >>> loss_fn = FamilyContrastiveLoss(temperature=0.07)
        >>> anchor = torch.randn(32, 768)      # batch of anchor embeddings
        >>> positive = torch.randn(32, 768)    # corresponding positives
        >>> negatives = torch.randn(32, 15, 768)  # 15 negatives per sample
        >>> loss = loss_fn(anchor, positive, negatives)
        >>> assert loss.requires_grad

    Hard Negative Types for Family Context:
        1. Same Person Different Event (SPDE):
           - Same family member appearing in different photos/contexts
           - Teaches model to learn identity-preserving features

        2. Temporal Neighbors:
           - Messages close in time but different semantic content
           - Prevents temporal shortcuts

        3. Similar Role Different Person:
           - E.g., different aunts, multiple grandchildren
           - Teaches fine-grained family role distinctions

    Reference:
        Based on InfoNCE (van den Oord et al., 2018) with adaptations
        for family relationship learning.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: Literal["mean", "sum", "none"] = "mean",
        hard_negative_weight: float = 1.0,
        use_hard_negatives: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.hard_negative_weight = hard_negative_weight
        self.use_hard_negatives = use_hard_negatives
        self.normalize = normalize

        # Learnable temperature (optional, for fine-tuning)
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log(), requires_grad=False)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor | None = None,
        hard_negative_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute family contrastive loss.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negatives: Optional explicit negatives [batch_size, num_negatives, embed_dim]
                       If None, uses in-batch negatives (other samples' positives)
            hard_negative_mask: Boolean mask indicating hard negatives
                               [batch_size, num_negatives] where True = hard negative

        Returns:
            Loss tensor (scalar if reduction != 'none')
        """
        batch_size = anchor.size(0)
        device = anchor.device

        # Normalize embeddings
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=-1)
            positive = F.normalize(positive, p=2, dim=-1)
            if negatives is not None:
                negatives = F.normalize(negatives, p=2, dim=-1)

        # Get temperature (use fixed or learned)
        if self.log_temperature.requires_grad:
            temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        else:
            temperature = self.temperature

        # Compute positive similarities
        # [batch_size]
        pos_sim = torch.sum(anchor * positive, dim=-1) / temperature

        if negatives is not None:
            # Use provided negatives
            # [batch_size, num_negatives]
            neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / temperature

            # Apply hard negative weighting if provided
            if hard_negative_mask is not None and self.use_hard_negatives:
                # Increase effective similarity for hard negatives (making them harder)
                hard_neg_boost = hard_negative_mask.float() * (self.hard_negative_weight - 1.0)
                neg_sim = neg_sim + hard_neg_boost

            # Concatenate positive and negatives for softmax
            # [batch_size, 1 + num_negatives]
            all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        else:
            # In-batch negatives: use all other positives as negatives
            # [batch_size, batch_size]
            all_positive_sim = torch.mm(anchor, positive.t()) / temperature

            # Mask out the positive (diagonal)
            mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            neg_sim = all_positive_sim.masked_fill(mask, float("-inf"))

            # [batch_size, batch_size] where column 0 is positive, rest are negatives
            all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
        # Equivalent to cross-entropy where positive is always label 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(all_sim, labels, reduction=self.reduction)

        return loss

    def forward_with_in_batch_negatives(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified forward using only in-batch negatives.

        This is more memory efficient when explicit negatives aren't needed.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]

        Returns:
            Loss tensor
        """
        return self.forward(anchor, positive, negatives=None)

    def forward_with_memory_bank(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        memory_bank: torch.Tensor,
        memory_hard_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward using a memory bank of cached negatives.

        This allows using more negatives than fit in a single batch
        by maintaining a queue of embeddings from previous batches.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            memory_bank: Cached negative embeddings [memory_size, embed_dim]
            memory_hard_mask: Optional mask for hard negatives in memory
                             [memory_size] where True = hard negative

        Returns:
            Loss tensor
        """
        batch_size = anchor.size(0)
        device = anchor.device

        # Normalize
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=-1)
            positive = F.normalize(positive, p=2, dim=-1)
            memory_bank = F.normalize(memory_bank, p=2, dim=-1)

        temperature = self.temperature

        # Positive similarities [batch_size]
        pos_sim = torch.sum(anchor * positive, dim=-1) / temperature

        # In-batch negative similarities [batch_size, batch_size]
        in_batch_sim = torch.mm(anchor, positive.t()) / temperature

        # Memory bank similarities [batch_size, memory_size]
        memory_sim = torch.mm(anchor, memory_bank.t()) / temperature

        # Apply hard negative weighting to memory
        if memory_hard_mask is not None and self.use_hard_negatives:
            hard_boost = memory_hard_mask.float() * (self.hard_negative_weight - 1.0)
            memory_sim = memory_sim + hard_boost.unsqueeze(0)

        # Mask diagonal in in-batch (these are positives)
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        in_batch_neg = in_batch_sim.masked_fill(mask, float("-inf"))

        # Combine: [batch_size, 1 + (batch_size-1) + memory_size]
        # But we simplify by keeping structure: [batch_size, 1 + batch_size + memory_size]
        all_sim = torch.cat([pos_sim.unsqueeze(-1), in_batch_neg, memory_sim], dim=-1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(all_sim, labels, reduction=self.reduction)

        return loss

    @staticmethod
    def mine_hard_negatives(
        anchor: torch.Tensor,
        candidate_negatives: torch.Tensor,
        num_hard: int = 5,
        strategy: Literal["hardest", "semi-hard", "random-hard"] = "semi-hard",
        margin: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mine hard negatives from a pool of candidates.

        Strategies:
            - hardest: Select negatives closest to anchor (may cause collapse)
            - semi-hard: Select negatives within margin of positive (recommended)
            - random-hard: Randomly sample from top-k hardest

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            candidate_negatives: Pool of candidate negatives [num_candidates, embed_dim]
            num_hard: Number of hard negatives to select per anchor
            strategy: Mining strategy
            margin: Margin for semi-hard mining

        Returns:
            Tuple of:
                - hard_negatives: Selected hard negatives [batch_size, num_hard, embed_dim]
                - hard_indices: Indices of selected negatives [batch_size, num_hard]
        """
        batch_size = anchor.size(0)
        num_candidates = candidate_negatives.size(0)
        device = anchor.device

        # Normalize for cosine similarity
        anchor_norm = F.normalize(anchor, p=2, dim=-1)
        candidates_norm = F.normalize(candidate_negatives, p=2, dim=-1)

        # Compute similarities [batch_size, num_candidates]
        similarities = torch.mm(anchor_norm, candidates_norm.t())

        if strategy == "hardest":
            # Select top-k most similar (hardest)
            _, indices = similarities.topk(num_hard, dim=-1, largest=True)

        elif strategy == "semi-hard":
            # Select negatives that are similar but not too similar
            # Semi-hard: sim < pos_sim but sim > pos_sim - margin
            # For simplicity, we select from the margin range
            upper = 1.0 - margin  # Not too close
            lower = -1.0  # Any dissimilar is ok

            valid_mask = (similarities < upper) & (similarities > lower)

            # For each anchor, randomly sample from valid negatives
            indices = torch.zeros(batch_size, num_hard, dtype=torch.long, device=device)

            for i in range(batch_size):
                valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_indices) >= num_hard:
                    # Prioritize harder ones (higher similarity among valid)
                    valid_sims = similarities[i, valid_indices]
                    _, top_valid = valid_sims.topk(
                        min(num_hard * 2, len(valid_indices)), largest=True
                    )
                    perm = torch.randperm(len(top_valid))[:num_hard]
                    indices[i] = valid_indices[top_valid[perm]]
                elif len(valid_indices) > 0:
                    # Repeat if not enough
                    repeats = (num_hard // len(valid_indices)) + 1
                    repeated = valid_indices.repeat(repeats)[:num_hard]
                    indices[i] = repeated
                else:
                    # Fallback to hardest if no valid
                    _, fallback = similarities[i].topk(num_hard, largest=True)
                    indices[i] = fallback

        elif strategy == "random-hard":
            # Randomly sample from top-k hardest
            top_k = min(num_hard * 3, num_candidates)
            _, top_indices = similarities.topk(top_k, dim=-1, largest=True)

            indices = torch.zeros(batch_size, num_hard, dtype=torch.long, device=device)
            for i in range(batch_size):
                perm = torch.randperm(top_k)[:num_hard]
                indices[i] = top_indices[i, perm]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Gather hard negatives
        # [batch_size, num_hard, embed_dim]
        hard_negatives = candidate_negatives[indices.flatten()].view(batch_size, num_hard, -1)

        return hard_negatives, indices

    @staticmethod
    def create_family_hard_negatives(
        embeddings: torch.Tensor,
        person_ids: torch.Tensor,
        event_ids: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        temporal_window: int = 5,
    ) -> dict[str, torch.Tensor]:
        """
        Create hard negative masks based on family context.

        This identifies which samples should be treated as hard negatives:
            1. Same person, different event (SPDE)
            2. Temporal neighbors within window
            3. Same event, different person

        Args:
            embeddings: Embedding matrix [num_samples, embed_dim]
            person_ids: Person identifier for each sample [num_samples]
            event_ids: Optional event identifier [num_samples]
            timestamps: Optional timestamps [num_samples]
            temporal_window: Window size for temporal neighbors

        Returns:
            Dictionary with hard negative masks:
                - 'spde_mask': Same person different event [num_samples, num_samples]
                - 'temporal_mask': Temporal neighbors [num_samples, num_samples]
                - 'combined_mask': Union of all hard negative types
        """
        num_samples = embeddings.size(0)
        device = embeddings.device

        # Initialize masks
        spde_mask = torch.zeros(num_samples, num_samples, dtype=torch.bool, device=device)
        temporal_mask = torch.zeros(num_samples, num_samples, dtype=torch.bool, device=device)

        # Same person, different event
        if event_ids is not None:
            person_match = person_ids.unsqueeze(0) == person_ids.unsqueeze(1)  # [N, N]
            event_diff = event_ids.unsqueeze(0) != event_ids.unsqueeze(1)  # [N, N]
            spde_mask = person_match & event_diff

        # Temporal neighbors
        if timestamps is not None:
            time_diff = torch.abs(timestamps.unsqueeze(0).float() - timestamps.unsqueeze(1).float())
            temporal_mask = (time_diff > 0) & (time_diff <= temporal_window)

        # Combined mask
        combined_mask = spde_mask | temporal_mask

        return {
            "spde_mask": spde_mask,
            "temporal_mask": temporal_mask,
            "combined_mask": combined_mask,
        }

    def enable_learned_temperature(self, requires_grad: bool = True) -> None:
        """Enable/disable learned temperature scaling."""
        self.log_temperature.requires_grad = requires_grad
        if requires_grad:
            # Use learned temperature in forward
            self._use_learned_temp = True
        else:
            self._use_learned_temp = False

    @property
    def learned_temperature(self) -> float:
        """Get current learned temperature value."""
        return self.log_temperature.exp().item()


# =============================================================================
# R-Drop Regularization
# =============================================================================


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks.

    R-Drop forces the model to produce consistent outputs across different
    dropout masks by adding a KL-divergence loss between two forward passes.

    This improves generalization by +1-2% on NLU tasks.

    Args:
        alpha: Weight for the KL-divergence regularization term. Default: 0.5
        reduction: Loss reduction method ('mean', 'sum', 'batchmean')

    Reference:
        Wu et al. "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021)

    Example:
        >>> rdrop = RDropLoss(alpha=0.5)
        >>> # Run model twice with different dropout
        >>> logits1 = model(input)
        >>> logits2 = model(input)  # Different dropout mask
        >>> ce_loss = F.cross_entropy(logits1, labels)
        >>> total_loss = rdrop(logits1, logits2, ce_loss)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        reduction: Literal["mean", "sum", "batchmean"] = "batchmean",
    ):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        ce_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute R-Drop loss combining CE loss with KL divergence.

        Args:
            logits1: First forward pass logits (batch_size, num_classes)
            logits2: Second forward pass logits (batch_size, num_classes)
            ce_loss: Cross-entropy loss from first pass

        Returns:
            Combined loss: CE + alpha * KL_divergence
        """
        # Compute KL divergence in both directions (symmetric)
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)

        p_soft = F.softmax(logits1, dim=-1)
        q_soft = F.softmax(logits2, dim=-1)

        # KL(P || Q) + KL(Q || P) for symmetric divergence
        kl_loss = F.kl_div(p, q_soft, reduction=self.reduction) + F.kl_div(
            q, p_soft, reduction=self.reduction
        )

        # Average the two directions
        kl_loss = kl_loss / 2.0

        # Combine with CE loss
        return ce_loss + self.alpha * kl_loss

    @staticmethod
    def compute_kl_divergence(
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        reduction: str = "batchmean",
    ) -> torch.Tensor:
        """Compute symmetric KL divergence between two distributions."""
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)
        p_soft = F.softmax(logits1, dim=-1)
        q_soft = F.softmax(logits2, dim=-1)

        kl_pq = F.kl_div(p, q_soft, reduction=reduction)
        kl_qp = F.kl_div(q, p_soft, reduction=reduction)

        return (kl_pq + kl_qp) / 2.0


# =============================================================================
# Adversarial Training - FGM (Fast Gradient Method)
# =============================================================================


class FGM:
    """
    Fast Gradient Method for adversarial training.

    FGM adds adversarial perturbations to word embeddings during training
    to improve model robustness. This typically gives +1-2% improvement.

    Args:
        model: The model to apply adversarial training to
        epsilon: Perturbation magnitude. Default: 1.0
        emb_name: Name of the embedding parameter to perturb. Default: 'word_embeddings'

    Reference:
        Miyato et al. "Adversarial Training Methods for Semi-Supervised Text Classification"

    Example:
        >>> fgm = FGM(model, epsilon=1.0)
        >>> # Normal forward + backward
        >>> loss = model(input, labels).loss
        >>> loss.backward()
        >>> # Adversarial attack
        >>> fgm.attack()
        >>> loss_adv = model(input, labels).loss
        >>> loss_adv.backward()
        >>> fgm.restore()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1.0,
        emb_name: str = "word_embeddings",
    ):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        """Add adversarial perturbation to embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                # Backup original embeddings
                self.backup[name] = param.data.clone()

                # Compute perturbation direction from gradients
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        # Perturb in gradient direction
                        r_at = self.epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self) -> None:
        """Restore original embeddings after adversarial step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class PGD:
    """
    Projected Gradient Descent for adversarial training.

    PGD is a stronger adversarial attack than FGM, using multiple
    smaller steps with projection back to epsilon-ball.

    Args:
        model: The model to apply adversarial training to
        epsilon: Maximum perturbation magnitude. Default: 1.0
        alpha: Step size for each iteration. Default: 0.3
        num_steps: Number of PGD steps. Default: 3
        emb_name: Name of the embedding parameter to perturb.

    Reference:
        Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks"

    Example:
        >>> pgd = PGD(model, epsilon=1.0, num_steps=3)
        >>> loss = model(input, labels).loss
        >>> loss.backward()
        >>> pgd.backup_grad()
        >>> for _ in range(pgd.num_steps):
        ...     pgd.attack(is_first=(t==0))
        ...     loss_adv = model(input, labels).loss
        ...     loss_adv.backward()
        >>> pgd.restore()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1.0,
        alpha: float = 0.3,
        num_steps: int = 3,
        emb_name: str = "word_embeddings",
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.emb_name = emb_name
        self.backup: dict[str, torch.Tensor] = {}
        self.grad_backup: dict[str, torch.Tensor] = {}

    def attack(self, is_first: bool = False) -> None:
        """
        Execute one step of PGD attack.

        Args:
            is_first: Whether this is the first attack step (backup embeddings)
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first:
                    # Backup original embeddings on first step
                    self.backup[name] = param.data.clone()

                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        # Take step in gradient direction
                        r_at = self.alpha * param.grad / norm
                        param.data.add_(r_at)

                        # Project back to epsilon-ball
                        param.data = self._project(
                            param.data,
                            self.backup[name],
                            self.epsilon,
                        )

    def _project(
        self,
        perturbed: torch.Tensor,
        original: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """Project perturbed embeddings back to epsilon-ball around original."""
        delta = perturbed - original
        norm = torch.norm(delta)
        if norm > epsilon:
            delta = delta * epsilon / norm
        return original + delta

    def restore(self) -> None:
        """Restore original embeddings after attack."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

    def backup_grad(self) -> None:
        """Backup gradients before adversarial steps."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self) -> None:
        """Restore original gradients after adversarial steps."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.grad_backup:
                    param.grad = self.grad_backup[name]


# =============================================================================
# Mixup Augmentation
# =============================================================================


class MixupLoss(nn.Module):
    """
    Mixup training for NLU tasks.

    Mixup creates virtual training examples by linearly interpolating
    between pairs of examples and their labels.

    For text, we apply mixup in the embedding space rather than input space.

    Args:
        alpha: Beta distribution parameter for mixing coefficient. Default: 0.4
        loss_fn: Base loss function to use. Default: CrossEntropyLoss

    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)

    Example:
        >>> mixup = MixupLoss(alpha=0.4)
        >>> embeddings, labels = batch
        >>> mixed_emb, labels_a, labels_b, lam = mixup.mixup_data(embeddings, labels)
        >>> outputs = model.classifier(mixed_emb)
        >>> loss = mixup(outputs, labels_a, labels_b, lam)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch of data.

        Args:
            x: Input features (batch_size, ...) - typically embeddings
            y: Labels (batch_size,)

        Returns:
            Tuple of (mixed_x, y_a, y_b, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def forward(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute mixup loss.

        Args:
            logits: Model predictions (batch_size, num_classes)
            labels_a: First set of labels
            labels_b: Second set of labels (shuffled)
            lam: Mixing coefficient

        Returns:
            Mixed loss value
        """
        loss_a = self.loss_fn(logits, labels_a)
        loss_b = self.loss_fn(logits, labels_b)
        return lam * loss_a + (1 - lam) * loss_b


class EmbeddingMixup(nn.Module):
    """
    Mixup applied in the embedding space for text classification.

    This is more suitable for NLU tasks than input-space mixup since
    we can't meaningfully interpolate between discrete tokens.

    Args:
        alpha: Beta distribution parameter. Default: 0.4
        apply_prob: Probability of applying mixup to a batch. Default: 0.5

    Example:
        >>> mixup = EmbeddingMixup(alpha=0.4)
        >>> embeddings = model.encoder(input_ids)
        >>> mixed_emb, targets = mixup(embeddings, labels)
        >>> logits = model.classifier(mixed_emb)
        >>> loss = F.cross_entropy(logits, targets)  # Soft targets
    """

    def __init__(
        self,
        alpha: float = 0.4,
        apply_prob: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.apply_prob = apply_prob

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply embedding mixup.

        Args:
            embeddings: Encoded representations (batch_size, hidden_size)
            labels: Class labels (batch_size,) or soft labels (batch_size, num_classes)
            num_classes: Number of classes (required if labels are hard)

        Returns:
            Tuple of (mixed_embeddings, mixed_soft_labels)
        """
        if not self.training or np.random.random() > self.apply_prob:
            # Convert to soft labels if needed
            if labels.dim() == 1 and num_classes is not None:
                soft_labels = F.one_hot(labels, num_classes).float()
            else:
                soft_labels = labels.float()
            return embeddings, soft_labels

        batch_size = embeddings.size(0)

        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Shuffle indices
        index = torch.randperm(batch_size, device=embeddings.device)

        # Mix embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]

        # Convert labels to soft labels if needed
        if labels.dim() == 1 and num_classes is not None:
            soft_labels = F.one_hot(labels, num_classes).float()
        else:
            soft_labels = labels.float()

        # Mix labels
        mixed_labels = lam * soft_labels + (1 - lam) * soft_labels[index]

        return mixed_embeddings, mixed_labels


# =============================================================================
# GlobalPointer Loss - Multi-Label Categorical Cross-Entropy for Span Detection
# =============================================================================


class GlobalPointerLoss(nn.Module):
    """
    Multi-Label Categorical Cross-Entropy Loss for GlobalPointer NER.

    This loss function is specifically designed for span-based NER using the
    GlobalPointer architecture (Su et al., 2022). It treats span detection as
    a multi-label classification problem where each (start, end) position can
    have multiple entity types.

    The key insight is using the logsumexp trick to compute a stable version
    of circle-loss style separation between positive and negative predictions.
    This naturally handles the extreme class imbalance in span detection where
    99%+ of positions are negative (non-entity spans).

    Mathematical Formulation:
        1. Flip sign for positive classes: y_pred = (1 - 2*y_true) * y_pred
        2. Mask opposite predictions with -inf
        3. Compute logsumexp for positive and negative separately
        4. Total loss = neg_loss + pos_loss

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
        mask_diagonal: Whether to exclude diagonal (single-token spans).
            Default False allows single-token entities.

    Reference:
        Su et al. "Global Pointer: Novel Efficient Span-based Approach
        for Named Entity Recognition" (arXiv:2208.03054)

    Example:
        >>> loss_fn = GlobalPointerLoss()
        >>> scores = torch.randn(2, 4, 128, 128)  # B, num_labels, L, L
        >>> labels = torch.zeros(2, 4, 128, 128)
        >>> labels[0, 0, 5, 10] = 1  # Entity from token 5 to 10
        >>> mask = torch.ones(2, 128)
        >>> loss = loss_fn(scores, labels, mask)
        >>> assert loss.requires_grad
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        mask_diagonal: bool = False,
    ):
        super().__init__()
        self.reduction = reduction
        self.mask_diagonal = mask_diagonal

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute GlobalPointer multi-label categorical cross-entropy loss.

        Args:
            y_pred: Span logits of shape (batch_size, num_labels, seq_len, seq_len).
                Raw logits (before sigmoid), where y_pred[b, l, i, j] is the
                score for entity type l spanning from token i to token j.
            y_true: Binary span labels of shape (batch_size, num_labels, seq_len, seq_len).
                1 indicates a true entity span, 0 otherwise.
            attention_mask: Optional padding mask of shape (batch_size, seq_len).
                1 for valid tokens, 0 for padding.

        Returns:
            Loss tensor. Shape depends on reduction:
                - 'mean' or 'sum': scalar
                - 'none': (batch_size * num_labels,)
        """
        batch_size, num_labels, seq_len, _ = y_pred.shape
        device = y_pred.device

        # Ensure y_true is on same device and float
        y_true = y_true.to(device).float()

        # Create upper triangular mask (valid spans only: start <= end)
        diagonal = 1 if self.mask_diagonal else 0
        triu_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=diagonal,
        )

        # Combine with padding mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            # Create 2D mask: position (i,j) is valid if both i and j are non-padding
            mask_i = attention_mask.unsqueeze(-1)  # (B, L, 1)
            mask_j = attention_mask.unsqueeze(-2)  # (B, 1, L)
            pad_mask = (mask_i * mask_j).bool()  # (B, L, L)
            # Combine with triu mask
            valid_mask = triu_mask.unsqueeze(0) & pad_mask  # (B, L, L)
        else:
            valid_mask = triu_mask.unsqueeze(0).expand(
                batch_size, seq_len, seq_len
            )  # (B, L, L)

        # Expand mask to include label dimension
        valid_mask = valid_mask.unsqueeze(1).expand(
            batch_size, num_labels, seq_len, seq_len
        )  # (B, num_labels, L, L)

        # Mask out invalid positions with large negative value
        # Use dtype-appropriate masking value to avoid overflow in float16
        mask_value = -1e4 if y_pred.dtype == torch.float16 else -1e12
        y_pred = y_pred.masked_fill(~valid_mask, mask_value)
        y_true = y_true.masked_fill(~valid_mask, 0.0)

        # Reshape to (batch_size * num_labels, seq_len * seq_len)
        y_pred_flat = y_pred.view(batch_size * num_labels, -1)
        y_true_flat = y_true.view(batch_size * num_labels, -1)

        # Compute multi-label categorical cross-entropy
        return self._multilabel_categorical_crossentropy(y_pred_flat, y_true_flat)

    def _multilabel_categorical_crossentropy(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core loss computation using logsumexp trick.

        This implements a circle-loss style separation where:
        - Positive samples are pushed to have HIGH scores
        - Negative samples are pushed to have LOW scores

        The logsumexp provides a smooth approximation to max, making the
        loss focus on the hardest examples (hardest negative and weakest positive).

        Args:
            y_pred: Flattened logits (batch_size * num_labels, seq_len * seq_len)
            y_true: Flattened labels (batch_size * num_labels, seq_len * seq_len)

        Returns:
            Loss tensor with specified reduction applied
        """
        # Use dtype-appropriate masking value to avoid overflow in float16
        mask_value = 1e4 if y_pred.dtype == torch.float16 else 1e12

        # Flip sign: positive classes get negative prediction, negative get positive
        # This inverts the optimization direction appropriately
        y_pred_adjusted = (1 - 2 * y_true) * y_pred

        # Mask out opposite class predictions with large negative value
        # y_pred_neg: predictions for negative samples (mask out positives)
        # y_pred_pos: predictions for positive samples (mask out negatives)
        y_pred_neg = y_pred_adjusted - y_true * mask_value  # Keep negatives, mask positives
        y_pred_pos = y_pred_adjusted - (1 - y_true) * mask_value  # Keep positives, mask negatives

        # Add a zero option for numerical stability when no positives/negatives exist
        # This prevents -inf from logsumexp of empty set
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        # LogSumExp aggregates scores in a soft-max-like fashion
        # - neg_loss: penalizes high scores on negative samples
        # - pos_loss: penalizes low scores on positive samples
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        # Combine losses
        loss = neg_loss + pos_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalGlobalPointerLoss(GlobalPointerLoss):
    """
    GlobalPointer loss with focal loss weighting for extreme imbalance.

    Extends GlobalPointerLoss by adding focal loss style (1-p)^gamma weighting
    to further down-weight easy examples. Useful when positive spans are
    extremely rare (e.g., <0.1% of positions).

    Note: In practice, the standard GlobalPointerLoss often works well enough
    due to the logsumexp aggregation naturally handling imbalance. Use this
    variant only if you observe the model predicting too many false positives.

    Args:
        gamma: Focal loss focusing parameter. Higher values more aggressively
            down-weight easy examples. Default: 2.0
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
        mask_diagonal: Whether to exclude diagonal (single-token spans)

    Example:
        >>> loss_fn = FocalGlobalPointerLoss(gamma=2.0)
        >>> scores = torch.randn(2, 4, 128, 128)
        >>> labels = torch.zeros(2, 4, 128, 128)
        >>> labels[0, 0, 5, 10] = 1
        >>> mask = torch.ones(2, 128)
        >>> loss = loss_fn(scores, labels, mask)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        mask_diagonal: bool = False,
    ):
        super().__init__(reduction=reduction, mask_diagonal=mask_diagonal)
        self.gamma = gamma

    def _multilabel_categorical_crossentropy(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal-weighted multi-label categorical cross-entropy.

        Applies (1-p)^gamma weighting to down-weight confident predictions.
        """
        # Use dtype-appropriate masking value to avoid overflow in float16
        mask_value = 1e4 if y_pred.dtype == torch.float16 else 1e12

        # Compute probabilities for focal weighting
        probs = torch.sigmoid(y_pred)

        # Focal weight: (1 - p_t)^gamma where p_t is prob of correct class
        # For positives: p_t = sigmoid(y_pred), weight = (1 - sigmoid(y_pred))^gamma
        # For negatives: p_t = 1 - sigmoid(y_pred), weight = sigmoid(y_pred)^gamma
        pt = y_true * probs + (1 - y_true) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Standard GlobalPointer loss computation
        y_pred_adjusted = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred_adjusted - y_true * mask_value
        y_pred_pos = y_pred_adjusted - (1 - y_true) * mask_value

        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        # Apply focal weighting (mean over sequence positions)
        loss = (neg_loss + pos_loss) * focal_weight.mean(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "MultipleNegativesRankingLoss",
    "CosineSimilarityLoss",
    "TripletLoss",
    "CRFLoss",
    "MultiTaskLoss",
    "UncertaintyWeightedLoss",
    "FamilyContrastiveLoss",
    "RDropLoss",
    "FGM",
    "PGD",
    "MixupLoss",
    "EmbeddingMixup",
    "GlobalPointerLoss",
    "FocalGlobalPointerLoss",
]
