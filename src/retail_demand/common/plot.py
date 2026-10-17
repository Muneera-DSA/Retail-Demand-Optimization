"""Shared figure style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GREY, LIGHT, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#d6d5d0", "#0b0b0b", "#52514e"


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def barh(ax, labels, values, colors, fmt="{:.2f}", errors=None):
    y = range(len(labels))
    ax.barh(list(y), values, color=colors, height=0.6, xerr=errors, ecolor=INK, capsize=2)
    ax.set_yticks(list(y), labels)
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK)
