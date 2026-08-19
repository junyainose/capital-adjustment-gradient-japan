"""Build the paper's figures.

    python build_figures.py

Writes PDF and PNG to output/figures/.  Requires the constructed panel; see
README for how to obtain the underlying data.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from build_tables import AXES, prepare   # noqa: E402
from src import gradients as gr          # noqa: E402

FIGURE_DIR = Path("output/figures")
LABELS = {"mrpk": "MRPK", "lev": "Leverage", "logk": r"log $k$"}


def save(fig, stem: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # The manuscript uses the vector version; the raster copy is for slides and
    # for previewing outside a TeX toolchain.  \includegraphics is called
    # without an extension, so pdflatex picks up the PDF whenever both exist.
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=150)
    plt.close(fig)
    return FIGURE_DIR / f"{stem}.pdf"


def figure_gradients(p, dep: str = "dlogk_h1", stem: str = "fig1_gradients"):
    """Year-by-year gradients for the three characteristics, with dispersion below.

    The top three panels show that only the MRPK gradient drifts.  The bottom
    panel shows that only leverage's dispersion moves, which is why the leverage
    gradient appears to drift when it is held in fixed units (Table 3).
    """
    years = sorted(p[gr.YEAR].unique())
    fit = gr.fit_year_gradients(p, dep, AXES, years)
    # The straight line is the drift estimated in Table 1, not a second fit
    # through the plotted points.  Fitting the points again would weight every
    # year equally, and the first and last years of the window are estimated
    # from far fewer firms than the rest; that alone moved the line by a quarter
    # in an earlier version and left the figure disagreeing with the table.
    trend = gr.fit_trend(p, dep, AXES)
    t_bar = p[gr.YEAR].mean()
    disp = gr.dispersion(p, ["leverage_lag", "mrpk_os_raw", "log_k_s_lag"])

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.4), sharex=True)
    for ax, key in zip(axes[:3], AXES):
        s = gr.gradient_series(fit, key, years)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.fill_between(s.year, s.coef - 1.645 * s.se, s.coef + 1.645 * s.se,
                        color="C0", alpha=0.18, lw=0)
        ax.plot(s.year, s.coef, color="C0", lw=1.8, marker="o", ms=3.5)
        level, _, _ = trend.get(f"level_{key}")
        slope, _, pval = trend.get(f"drift_{key}")
        ax.plot(s.year, level + slope * (s.year - t_bar),
                color="C3", ls="--", lw=1.2)
        ax.set_ylabel(r"$\delta_t$")
        ax.grid(alpha=0.25)
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        ax.set_title(f"{LABELS[key]}   (drift {10 * slope:+.2f} per decade{stars})",
                     fontsize=10, loc="left")

    shown = gr.gradient_series(fit, "mrpk", years).year.tolist()
    index = disp.loc[shown] / disp.loc[shown].iloc[0] * 100
    for col, name, marker in [("leverage_lag", "Leverage", "o"),
                              ("mrpk_os_raw", "MRPK", "s"),
                              ("log_k_s_lag", r"log $k$", "^")]:
        axes[3].plot(index.index, index[col], lw=1.8, marker=marker, ms=3.5,
                     label=name)
    axes[3].axhline(100, color="0.6", lw=0.8)
    axes[3].set_ylabel("index, first year = 100")
    axes[3].set_xlabel("fiscal year")
    axes[3].legend(fontsize=8, frameon=False)
    axes[3].grid(alpha=0.25)
    axes[3].set_title("Within industry-year dispersion of each characteristic",
                      fontsize=10, loc="left")

    fig.suptitle("Investment gradients and cross-sectional dispersion\n"
                 r"(dependent variable: $100\times\Delta\log k$, $h=1$)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save(fig, stem)


def main(panel_path: str | None = None) -> None:
    p = prepare(panel_path) if panel_path else prepare()
    print(f"  wrote {figure_gradients(p)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=None,
                    help="path to the constructed panel; pass the synthetic "
                         "panel to test the code without a NEEDS licence")
    main(ap.parse_args().panel)
