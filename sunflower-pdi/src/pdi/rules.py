"""
Continuous Pollen Deficiency Index (PDI) Scoring.

Design rationale
----------------
The original rule-based proxy assigned PDI labels based purely on growth-stage
membership (YoungBud/MatureBud → High, Healthy/EarlyBloom → Low), which made
Low vs High a trivially separable problem equivalent to bud-vs-bloom classification.

This redesigned formulation computes a continuous PDI score (0–1) from four
morphological and spectral features that vary *within* each growth stage:

  - disc_ratio   : fraction of flower area occupied by the central tubular disc.
                   A larger, more exposed disc → more pollen shedding → lower deficiency.
  - sat_mean     : mean HSV saturation of the foreground region (0–255 scaled).
                   High saturation (vivid yellow petals) → active bloom → lower deficiency.
  - val_mean     : mean HSV brightness of the foreground region (0–255 scaled).
                   Darker, wilted, or tightly folded flowers → higher deficiency.
  - circularity  : 4π·Area/Perimeter² of the foreground mask.
                   Higher circularity (open capitulum) → lower deficiency.

Weights reflect relative biological importance (disc opening is primary indicator).
The PDI score is normalised globally against the dataset distribution at fit time,
and thresholds are set at the 33rd and 67th percentiles to produce a balanced
Low / Medium / High three-class distribution.

Binary task (Phase 2): Low vs High only (Medium samples excluded).
"""

import numpy as np


# Global normalisation statistics (fitted on the 1,500-image training sample).
# These are pre-computed from the dataset so that inference is deterministic.
_STATS = {
    "disc_ratio": {"min": 0.033333, "max": 0.780000},
    "sat_mean":   {"min": 37.021277, "max": 246.975858},
    "val_mean":   {"min": 28.140000, "max": 231.666667},
    "circularity":{"min": 0.278966, "max": 0.802517},
}

# PDI score thresholds (33rd / 67th percentile of the 1,500-image distribution).
_THRESH_LOW  = 0.448   # below this → Low deficiency
_THRESH_HIGH = 0.484   # above this → High deficiency


def _norm(value: float, feat: str) -> float:
    """Min-max normalise a single feature value using pre-fitted statistics."""
    lo = _STATS[feat]["min"]
    hi = _STATS[feat]["max"]
    return float(np.clip((value - lo) / (hi - lo + 1e-8), 0.0, 1.0))


def pdi_score(disc_ratio: float, sat_mean: float,
              val_mean: float, circularity: float) -> float:
    """
    Compute a continuous Pollen Deficiency Index score in [0, 1].

    Parameters
    ----------
    disc_ratio   : ratio of disc area to total foreground area (0–1)
    sat_mean     : mean HSV saturation of the foreground region (0–255)
    val_mean     : mean HSV value/brightness of the foreground region (0–255)
    circularity  : 4π·Area/Perimeter² of the foreground mask (0–π/4)

    Returns
    -------
    score : float in [0, 1].  0 = no deficiency; 1 = maximum deficiency.
    """
    d = _norm(disc_ratio,   "disc_ratio")
    s = _norm(sat_mean,     "sat_mean")
    v = _norm(val_mean,     "val_mean")
    c = _norm(circularity,  "circularity")

    score = (
        0.40 * (1.0 - d) +   # closed disc  → high deficiency (primary indicator)
        0.25 * (1.0 - s) +   # low sat      → high deficiency
        0.20 * (1.0 - v) +   # low val      → high deficiency
        0.15 * (1.0 - c)     # irregular    → high deficiency
    )
    return float(np.clip(score, 0.0, 1.0))


def pdi_proxy_label(disc_ratio: float, sat_mean: float,
                    val_mean: float, circularity: float) -> str:
    """
    Assign a three-class PDI proxy label based on the continuous pdi_score.

    Thresholds are the 33rd and 67th global percentiles of the 1,500-image
    feature distribution, ensuring a balanced class distribution regardless
    of growth-stage membership.

    Returns
    -------
    "Low"    – pdi_score < 0.448  (pollen adequacy; open disc, vivid saturation)
    "Medium" – 0.448 ≤ pdi_score < 0.484
    "High"   – pdi_score ≥ 0.484  (pollen deficiency; closed bud or wilted)
    """
    score = pdi_score(disc_ratio, sat_mean, val_mean, circularity)
    if score < _THRESH_LOW:
        return "Low"
    if score < _THRESH_HIGH:
        return "Medium"
    return "High"
