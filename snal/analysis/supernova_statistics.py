"""
Annual stacked bar plot from Rochester snstats.csv.
https://www.rochesterastronomy.org/snimages/snstats.csv
"""
#%%
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SNSTATS_URL = "https://www.rochesterastronomy.org/snimages/snstats.csv"
YEAR_LO, YEAR_HI = 2010, 2025

# Bottom → top in the stack. One color each: Ia, Ib, Ic, II, Unclassified (red).
STACK: list[tuple[str, list[str], str]] = [
    ("Ia", ["Ia", "Ia-02cx", "Ia-91bg", "Ia-91T", "Ia-pec"], "#1565C0"),
    ("Ib", ["Ib", "Ibn", "Ib-pec"], "#F57C00"),
    ("Ic", ["Ic", "Icn", "Ic-pec"], "#2E7D32"),
    ("II", ["II", "IIn", "IIP", "IIb", "IIL", "SLSN-II", "II-pec"], "#6A1B9A"),
    ("Unclassified", ["I", "SLSN-I", "LBV", "nonSN", "untyped"], "#C62828"),
]


def load_snstats(url: str = SNSTATS_URL) -> pd.DataFrame:
    with urlopen(url) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(text), on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    y = pd.to_numeric(df["Year"], errors="coerce")
    df = df[y.notna()].copy()
    df["Year"] = y[y.notna()].astype(int)
    df = df.sort_values("Year").reset_index(drop=True)
    for c in df.columns:
        if c != "Year":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _sum_cols(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    use = [c for c in names if c in df.columns]
    if not use:
        return np.zeros(len(df))
    return df[use].sum(axis=1).to_numpy()


def plot_annual_bars(df: pd.DataFrame, out_path: Path) -> None:
    df = df[(df["Year"] >= YEAR_LO) & (df["Year"] <= YEAR_HI)].copy()
    years = df["Year"].to_numpy()
    layers = np.vstack([_sum_cols(df, cols) for _label, cols, _c in STACK])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.85
    bottom = np.zeros(len(years))
    for i, (label, _cols, color) in enumerate(STACK):
        ax.bar(
            years,
            layers[i],
            width=w,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.35,
        )
        bottom += layers[i]

    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Count", fontsize=18)
    ax.set_title(
        "Supernova discoveries by spectroscopic class (Rochester snstats)",
        fontsize=20,
    )
    ax.set_xlim(YEAR_LO - 0.5, YEAR_HI + 0.5)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(loc="upper left", ncol=3, fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Annual SN class stacked bar plot from Rochester CSV")
    p.add_argument("--url", default=SNSTATS_URL)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or Path(__file__).resolve().parent / "snstats_annual.png"
    plot_annual_bars(load_snstats(args.url), out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

# %%
