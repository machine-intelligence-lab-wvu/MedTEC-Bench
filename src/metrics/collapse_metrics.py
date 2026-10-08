"""
Metrics for quantifying text embedding collapse.
"""

import numpy as np
from typing import Dict, List
from scipy.stats import skew


def discrimination_score(pos_sims: np.ndarray, neg_sims: np.ndarray) -> float:
    """
    DS = cos(img, text_pos) - max(cos(img, text_neg_i))
    Should be >> 0. If <= 0, model prefers wrong text.
    """
    return float(np.max(pos_sims) - np.max(neg_sims))


def collapse_ratio(pos_sims: np.ndarray, neg_sims: np.ndarray) -> float:
    """
    CR = mean(cos(img, text_neg_i)) / cos(img, text_pos)
    Should be << 1. Close to 1 = collapse. Above 1 = complete failure.
    """
    best_pos = np.max(pos_sims)
    if abs(best_pos) < 1e-8:
        return float("inf")
    return float(np.mean(neg_sims) / best_pos)


def embedding_isotropy(text_embeddings: np.ndarray) -> float:
    """
    Ratio of min/max eigenvalues of text embedding covariance matrix.
    Low isotropy -> embeddings cluster in narrow cone -> collapse.
    """
    cov = np.cov(text_embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues.min() / eigenvalues.max())


def effective_dimensionality(text_embeddings: np.ndarray) -> float:
    """
    Participation-ratio effective dimensionality (Roy & Vetterli, 2007):

        D_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2

    Equals the ambient dimension D when all eigenvalues are equal (fully isotropic)
    and drops toward 1 as mass concentrates in a few directions. Complements the
    min/max ratio isotropy by measuring how many directions carry variance, not
    the worst-case contrast.
    """
    cov = np.cov(text_embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 0.0
    s1 = float(eigenvalues.sum())
    s2 = float((eigenvalues ** 2).sum())
    if s2 < 1e-20:
        return 0.0
    return (s1 * s1) / s2


def hubness_score(similarity_matrix: np.ndarray, k: int = 5) -> float:
    """
    Skewness of k-occurrence distribution.
    High hubness = "hub" texts spuriously match many images.
    """
    # For each text, count how many images have it in their top-k
    n_images, n_texts = similarity_matrix.shape
    k_occurrences = np.zeros(n_texts)

    for i in range(n_images):
        top_k = np.argsort(similarity_matrix[i])[-k:]
        k_occurrences[top_k] += 1

    return float(skew(k_occurrences))


def evaluate_probe(
    image_embedding: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a single probe and return all collapse metrics.
    All embeddings should be L2-normalized.
    """
    pos_sims = image_embedding @ positive_embeddings.T
    neg_sims = image_embedding @ negative_embeddings.T

    pos_sims = pos_sims.flatten()
    neg_sims = neg_sims.flatten()

    return {
        "discrimination_score": discrimination_score(pos_sims, neg_sims),
        "collapse_ratio": collapse_ratio(pos_sims, neg_sims),
        "best_positive_sim": float(np.max(pos_sims)),
        "best_negative_sim": float(np.max(neg_sims)),
        "mean_negative_sim": float(np.mean(neg_sims)),
        "sim_range": float(np.max(pos_sims) - np.min(neg_sims)),
    }
