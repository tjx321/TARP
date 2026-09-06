"""Re-centered Fisher Linear Discriminant (FLD) projection for TARP."""

import torch

from utils import compute_class_means_with_text_prior


def fld_center(feat, label, num_classes, text_prototypes, text_prior_tau=4.0):
    """FLD projection with text-prior re-centered class means.

    The between-class scatter is built from class means that are re-centered
    towards the CLIP text prototypes with strength ``text_prior_tau`` (tau,
    fixed to 4 across all datasets and shots).

    Args:
        feat:             [N, D] support/cache features.
        label:            [N] integer class labels aligned with ``feat``.
        num_classes:      number of classes C.
        text_prototypes:  [C, D] CLIP text weights (row = class).
        text_prior_tau:   text-prior strength (tau, default 4.0).

    Returns:
        W: [D, C - 1] FLD projection matrix.
    """
    device = feat.device
    mean_total = torch.mean(feat, dim=0)

    _, class_sizes, _, class_means, _ = compute_class_means_with_text_prior(
        feat,
        label,
        num_classes,
        text_prototypes,
        text_prior_tau=text_prior_tau,
    )

    mean_diff = feat - class_means[label]
    mean_diff_total = class_means - mean_total
    # Between-class scatter S_B from the re-centered class means.
    S_B = (mean_diff_total.T @ (mean_diff_total * class_sizes.unsqueeze(1))).float()

    # Within-class scatter under the same-covariance hypothesis:
    # regularized pseudo-inverse of the covariance.
    S_W_inv = num_classes * mean_diff.shape[1] * torch.linalg.pinv(
        (mean_diff.shape[0] - 1) * mean_diff.T.cov()
        + mean_diff.T.cov().trace() * torch.eye(mean_diff.shape[1], device=device)
    )

    eig_vals, eig_vecs = torch.linalg.eig(S_W_inv @ S_B)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real

    sorted_indices = torch.argsort(eig_vals, descending=True)
    W = eig_vecs[:, sorted_indices[:num_classes - 1]]
    return W
