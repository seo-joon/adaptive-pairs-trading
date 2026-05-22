from __future__ import annotations

from typing import Dict


def filter_signal(features: Dict) -> bool:
    return True


def position_sizer(confidence: float, context: Dict) -> Dict:
    return {}


def compute_features(context: Dict) -> Dict:
    return {}


class RegimeModel:
    def fit(self, *args, **kwargs):
        return self

    def predict(self, *args, **kwargs):
        return None

    def save(self, path: str):
        pass

    def load(self, path: str):
        return self


