"""
SERENOVA - Unified Feature Representation
"""

from __future__ import annotations


def build_feature_vector(
    maternal_features: dict,
    fetal_features: dict,
) -> dict:

    return {
        "maternal": maternal_features,
        "fetal": fetal_features,
    }