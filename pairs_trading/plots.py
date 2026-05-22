from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(equity: pd.Series, path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


