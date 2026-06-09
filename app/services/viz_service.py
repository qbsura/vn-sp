"""
app/services/viz_service.py
============================
Tái hiện các figures của bài báo Li et al. (Engineering Applications of AI, 2026)
bằng matplotlib/seaborn.

Mỗi hàm trả về matplotlib.figure.Figure object.
Caller (app/api/viz.py) sẽ dùng _fig_to_base64() convert sang base64 PNG.

Color scheme: dark theme khớp với frontend dashboard (#0f0f1a bg, #64b5f6 blue, #e94560 red)

Figures:
  fig1_pipeline_framework()                         → Fig. 1 (static diagram)
  fig2_deviation_scatter(df, ticker)                → Fig. 2
  fig3_feature_distributions(df_wavelet, ticker)    → Fig. 3
  fig4_scaling_flowchart()                          → Fig. 4 (static diagram)
  fig5_wavelet_decomposition(df, ticker)            → Fig. 5
  fig6_wavelet_functions()                          → Fig. 6 (static, pywt db4)
  fig7_approx_coefficients(df_wavelet, ticker)      → Fig. 7
  fig8_detail_coefficients(df_wavelet, ticker)      → Fig. 8
  fig9_level1_decomposition()                       → Fig. 9 (static diagram)
  fig10_correlation_matrix(df_wavelet, ticker)      → Fig. 10
  fig11_mse_comparison(ticker, currency)            → Fig. 11 (reads eval results)

Tham chiếu:
  Li et al., EAAI 165 (2026) 113390, Figures 1–11.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; phải gọi trước import pyplot
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# DARK THEME CONSTANTS
# =============================================================================

# Màu dashboard (match frontend/css/style.css variables)
C_BG      = "#0f0f1a"   # deep dark background
C_AX      = "#0a0a15"   # axes background
C_EDGE    = "#2a2a4a"   # axes border / spine
C_FG      = "#e0e0e0"   # text / foreground
C_BLUE    = "#64b5f6"   # primary blue
C_RED     = "#e94560"   # accent red
C_GREEN   = "#4caf50"   # green
C_ORANGE  = "#ff9800"   # orange
C_PURPLE  = "#9c27b0"   # purple
C_TEAL    = "#26c6da"   # teal
C_YELLOW  = "#ffeb3b"   # yellow
C_GRID    = "#1e1e3a"   # subtle grid lines

# Palette cho multi-series plots (5 models)
MODEL_PALETTE = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE]

# rcParams context để apply dark theme nhất quán
DARK_RC = {
    "figure.facecolor"  : C_BG,
    "axes.facecolor"    : C_AX,
    "axes.edgecolor"    : C_EDGE,
    "axes.labelcolor"   : C_FG,
    "axes.titlecolor"   : C_FG,
    "xtick.color"       : C_FG,
    "ytick.color"       : C_FG,
    "text.color"        : C_FG,
    "legend.facecolor"  : "#0f0f2a",
    "legend.edgecolor"  : C_EDGE,
    "grid.color"        : C_GRID,
    "grid.alpha"        : 0.4,
    "font.size"         : 10,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
}


# =============================================================================
# FIG. 1 — Pipeline Framework (static diagram)
# =============================================================================

def fig1_pipeline_framework() -> plt.Figure:
    """
    Tái hiện Fig. 1: Kiến trúc pipeline đầy đủ của VNSP.

    Stock Data → Feature Derivation → Wavelet Decomp (A1/D1)
    → Feature Selection → Scaling → Sequence Window
    → BiLSTM (dual-branch) → Regression / Classification Output

    Returns:
        matplotlib.Figure
    """
    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 5)
        ax.axis("off")

        # ── Single-lane pipeline boxes ─────────────────────────────────────
        #   (label, x_center, y_center, width, height, facecolor)
        boxes = [
            ("Stock Data\n(OHLCV)", 1.2, 2.5, 1.8, 1.2, C_BLUE),
            ("Deviation\nFeature\n(Close−Open)", 3.4, 2.5, 1.8, 1.2, C_BLUE),
            ("SWT db4\nLevel-1", 5.6, 2.5, 1.8, 1.2, C_TEAL),
            ("Feature\nSelection\n(corr < 0.95)", 7.8, 2.5, 1.8, 1.2, C_GREEN),
            ("Feature\nScaling\n(Std / Robust)", 10.0, 2.5, 1.8, 1.2, C_GREEN),
            ("Sequence\nWindow\n(slide)", 12.2, 2.5, 1.8, 1.2, C_ORANGE),
        ]

        for label, xc, yc, w, h, color in boxes:
            _draw_box(ax, xc, yc, w, h, label, color, fontsize=8)

        # ── BiLSTM dual-branch section ─────────────────────────────────────
        # A1 branch (top)
        _draw_box(ax, 14.1, 3.6, 1.6, 0.85, "BiLSTM\n(A1 branch)", C_RED, fontsize=8)
        # D1 branch (bottom)
        _draw_box(ax, 14.1, 1.5, 1.6, 0.85, "BiLSTM\n(D1 branch)", C_RED, fontsize=8)
        # Output box
        _draw_box(ax, 15.8, 2.55, 0.3, 1.8, "↓\n↓", "#444466", fontsize=7)  # concat marker

        # ── Arrows (single lane) ──────────────────────────────────────────
        arrow_kw = dict(arrowstyle="->", color=C_FG, lw=1.5,
                        connectionstyle="arc3,rad=0")
        arrow_pts = [
            (2.1, 2.5,  2.5, 2.5),    # Stock Data → Deviation
            (4.3, 2.5,  4.7, 2.5),    # Deviation → SWT
            (6.5, 2.5,  6.9, 2.5),    # SWT → Feature Sel
            (8.7, 2.5,  9.1, 2.5),    # Feature Sel → Scaling
            (10.9, 2.5, 11.3, 2.5),   # Scaling → Window
        ]
        for x1, y1, x2, y2 in arrow_pts:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=C_FG, lw=1.5))

        # Window → bifurcate to A1 and D1 branches
        ax.annotate("", xy=(13.3, 3.6), xytext=(13.1, 2.5),
                    arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.5))
        ax.annotate("", xy=(13.3, 1.5), xytext=(13.1, 2.5),
                    arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.5))
        ax.text(13.1, 2.5, "A1\nD1", ha="left", va="center",
                fontsize=7, color=C_TEAL, fontweight="bold")

        # BiLSTM branches → concat
        ax.annotate("", xy=(14.92, 3.6), xytext=(15.2, 3.0),
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
        ax.annotate("", xy=(14.92, 1.5), xytext=(15.2, 2.1),
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

        # Output labels
        ax.text(15.6, 4.2, "Regression\nOutput", ha="center", va="center",
                fontsize=8, color=C_ORANGE,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a3a",
                          edgecolor=C_ORANGE, alpha=0.9))
        ax.text(15.6, 0.8, "Classification\nOutput", ha="center", va="center",
                fontsize=8, color=C_GREEN,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a3a",
                          edgecolor=C_GREEN, alpha=0.9))
        ax.annotate("", xy=(15.35, 4.0), xytext=(15.45, 3.55),
                    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2))
        ax.annotate("", xy=(15.35, 1.0), xytext=(15.45, 1.45),
                    arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2))

        ax.set_title(
            "VNSP Pipeline Framework — Li et al. (EAAI 2026), Fig. 1",
            fontsize=13, color=C_FG, pad=14,
        )
        plt.tight_layout()
    return fig


def _draw_box(ax, xc, yc, w, h, label, color, fontsize=9):
    """Helper: draws a rounded box centered at (xc, yc) with text."""
    box = mpatches.FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, alpha=0.85,
        edgecolor="white", linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(xc, yc, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold",
            multialignment="center", zorder=4)


# =============================================================================
# FIG. 2 — Deviation Scatter Plot
# =============================================================================

def fig2_deviation_scatter(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Tái hiện Fig. 2: Scatter plot Close Price vs Deviation.

    Cho thấy Deviation (Close - Open) tăng biến động khi giá tăng,
    phản ánh áp lực mua/bán mạnh hơn ở mức giá cao.

    Args:
        df:     DataFrame với cột 'Close' và 'Deviation' (hoặc tính lại).
        ticker: Mã cổ phiếu (VCB / VIC).

    Returns:
        matplotlib.Figure
    """
    # Tính Deviation nếu chưa có
    if "Deviation" not in df.columns:
        if "Close" in df.columns and "Open" in df.columns:
            df = df.copy()
            df["Deviation"] = df["Close"] - df["Open"]
        else:
            raise ValueError("df thiếu cột 'Deviation' hoặc ('Close', 'Open')")

    df_clean = df[["Close", "Deviation"]].dropna()

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            df_clean["Close"], df_clean["Deviation"],
            alpha=0.35, s=8,
            color=C_BLUE, edgecolors="none",
        )

        # Đường zero (Deviation = 0) làm mốc tham chiếu
        ax.axhline(y=0, color=C_RED, linewidth=1.2, linestyle="--",
                   alpha=0.8, label="Deviation = 0")

        # Trend line (dùng numpy polyfit bậc 1)
        x_vals = df_clean["Close"].values
        y_vals = df_clean["Deviation"].values
        try:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            ax.plot(x_line, p(x_line), color=C_TEAL, linewidth=1.5,
                    linestyle=":", alpha=0.7, label="Trend")
        except Exception:
            pass  # bỏ qua nếu polyfit lỗi

        ax.set_title(
            f"{ticker} — Deviation Changes as Stock Price Increases",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Close Price", fontsize=11)
        ax.set_ylabel("Deviation (Close − Open)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    return fig


# =============================================================================
# FIG. 3 — Feature Distributions (Histogram + KDE)
# =============================================================================

def fig3_feature_distributions(df_wavelet: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Tái hiện Fig. 3: Histogram + KDE của các wavelet features.

    Cho thấy bimodal distribution do giá tăng theo thời gian.
    Plot tất cả Approx + Detail columns có trong df_wavelet (tối đa 6).

    Args:
        df_wavelet: DataFrame với wavelet features (e.g., Open_Approx, Volume_Detail...).
        ticker:     Mã cổ phiếu.

    Returns:
        matplotlib.Figure (2×n grid)
    """
    # Ưu tiên các features biểu diễn tốt nhất theo thứ tự bài báo
    preferred_order = [
        "Open_Approx", "Open_Detail",
        "Volume_Approx", "Volume_Detail",
        "Deviation_Approx", "Deviation_Detail",
        "Low_Approx", "Low_Detail",
        "High_Approx", "High_Detail",
    ]
    # Lấy wavelet columns có trong df
    avail_cols = [c for c in preferred_order if c in df_wavelet.columns]
    # Fallback: any column ending with _Approx / _Detail
    if not avail_cols:
        avail_cols = [c for c in df_wavelet.columns
                      if c.endswith("_Approx") or c.endswith("_Detail")]
    # Giới hạn 6 panels
    cols_to_plot = avail_cols[:6]

    if not cols_to_plot:
        # Return empty figure nếu không có wavelet features
        with plt.rc_context(DARK_RC):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Không có wavelet features trong dataset",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color=C_FG)
            ax.axis("off")
        return fig

    n = len(cols_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(DARK_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            data = df_wavelet[col].dropna().values

            # Histogram
            ax.hist(data, bins=50, density=True, alpha=0.55,
                    color=C_GREEN, edgecolor="none", label="Histogram")

            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data, bw_method="scott")
                x_kde = np.linspace(data.min(), data.max(), 300)
                ax.plot(x_kde, kde(x_kde), color=C_TEAL, linewidth=2.0, label="KDE")
            except Exception:
                pass

            # Formatting
            suffix = " (Approx)" if col.endswith("_Approx") else " (Detail)"
            feature_base = col.replace("_Approx", "").replace("_Detail", "")
            ax.set_title(f"{feature_base}{suffix}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))

        # Ẩn axes thừa
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"{ticker} — Wavelet Feature Distributions (Fig. 3)\n"
            "Bimodal pattern do tăng trưởng giá theo thời gian",
            fontsize=12, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
    return fig


# =============================================================================
# FIG. 4 — Scaling Flowchart (static diagram)
# =============================================================================

def fig4_scaling_flowchart() -> plt.Figure:
    """
    Tái hiện Fig. 4: Flowchart quy trình Feature Scaling.

    Phân loại features theo phân phối:
      Chuẩn (normal) → Standard Scaler
      Lệch/Outliers  → Robust Scaler
      Open_Approx    → No Scale (giữ trend information)

    Returns:
        matplotlib.Figure
    """
    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 7)
        ax.axis("off")

        # ── Boxes ─────────────────────────────────────────────────────────
        # Input
        _draw_box(ax, 6.0, 6.2, 3.0, 0.9, "Raw Wavelet Features\n(10 columns)", C_BLUE, fontsize=9)

        # Decision diamond (dùng polygon)
        _draw_diamond(ax, 6.0, 4.8, 2.5, 0.9,
                      "Check Distribution", C_ORANGE, fontsize=9)

        # Three branches
        _draw_box(ax, 1.8, 3.1, 2.8, 1.0,
                  "Normal Distribution?\n→ Standard Scaler\n(Open_Detail, High_Detail,\nLow_Detail, Dev_Approx, Dev_Detail)",
                  C_GREEN, fontsize=7.5)
        _draw_box(ax, 6.0, 3.1, 2.8, 1.0,
                  "Skewed / Outliers?\n→ Robust Scaler\n(High_Approx, Low_Approx,\nVolume_Approx, Volume_Detail)",
                  C_RED, fontsize=7.5)
        _draw_box(ax, 10.2, 3.1, 2.8, 1.0,
                  "Trend Feature?\n→ No Scale\n(Open_Approx only)",
                  C_PURPLE, fontsize=7.5)

        # Output
        _draw_box(ax, 6.0, 1.4, 3.0, 0.9,
                  "Scaled Feature Matrix\n(ready for sequence builder)", C_TEAL, fontsize=9)

        # ── Arrows ────────────────────────────────────────────────────────
        _arrow(ax, 6.0, 5.75, 6.0, 5.25)           # Input → Decision
        _arrow(ax, 4.75, 4.8, 1.8, 3.6)             # Decision → Standard
        _arrow(ax, 6.0, 4.35, 6.0, 3.6)             # Decision → Robust
        _arrow(ax, 7.25, 4.8, 10.2, 3.6)            # Decision → No Scale
        _arrow(ax, 1.8, 2.6, 6.0, 1.85, color=C_GREEN)   # Standard → Output
        _arrow(ax, 6.0, 2.6, 6.0, 1.85)             # Robust → Output
        _arrow(ax, 10.2, 2.6, 6.0, 1.85, color=C_PURPLE)  # NoScale → Output

        # ── Labels ────────────────────────────────────────────────────────
        ax.text(3.3, 4.9, "Normal?", fontsize=8, color=C_GREEN, ha="center")
        ax.text(6.0, 4.1, "Skewed?", fontsize=8, color=C_RED, ha="center")
        ax.text(8.7, 4.9, "No Scale?", fontsize=8, color=C_PURPLE, ha="center")

        ax.set_title("Feature Scaling Flowchart — VNSP (Fig. 4)",
                     fontsize=13, color=C_FG, pad=14)
        plt.tight_layout()
    return fig


def _draw_diamond(ax, xc, yc, w, h, label, color, fontsize=9):
    """Draw a diamond (decision) shape."""
    # Diamond corners: top, right, bottom, left
    xs = [xc,      xc + w/2, xc,       xc - w/2, xc]
    ys = [yc + h/2, yc,       yc - h/2, yc,        yc + h/2]
    poly = plt.Polygon(list(zip(xs, ys)),
                       facecolor=color, alpha=0.8,
                       edgecolor="white", linewidth=1.2, zorder=3)
    ax.add_patch(poly)
    ax.text(xc, yc, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold",
            multialignment="center", zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=None):
    """Draw a simple arrow from (x1,y1) to (x2,y2)."""
    c = color or C_FG
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5))


# =============================================================================
# FIG. 5 — Wavelet Decomposition Visualization (Deviation over time)
# =============================================================================

def fig5_wavelet_decomposition(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Tái hiện Fig. 5: Time series của Deviation để minh họa wavelet decomposition.

    Hiển thị tín hiệu gốc + (nếu có) Approx + Detail decomposition.

    Args:
        df:     DataFrame với DatetimeIndex và cột 'Deviation' (hoặc 'Close').
        ticker: Mã cổ phiếu.

    Returns:
        matplotlib.Figure
    """
    with plt.rc_context(DARK_RC):
        # Xác định cột cần plot
        has_approx = "Deviation_Approx" in df.columns
        has_detail = "Deviation_Detail" in df.columns
        has_raw    = "Deviation" in df.columns

        if has_approx and has_detail:
            # 3-panel: original + approx + detail (nếu có Deviation)
            n_panels = 3 if has_raw else 2
            fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels),
                                     sharex=True)

            panel_i = 0
            if has_raw:
                axes[panel_i].plot(df.index, df["Deviation"],
                                   color=C_BLUE, linewidth=0.8, alpha=0.9)
                axes[panel_i].set_ylabel("Deviation (original)", fontsize=9)
                axes[panel_i].set_title(
                    f"{ticker} — Wavelet Decomposition of Deviation (Fig. 5)",
                    fontsize=13)
                axes[panel_i].grid(True, alpha=0.3)
                panel_i += 1

            axes[panel_i].plot(df.index, df["Deviation_Approx"],
                               color=C_GREEN, linewidth=1.0)
            axes[panel_i].set_ylabel("A1 (Approx)", fontsize=9)
            axes[panel_i].set_title("Approximation Coefficients (Low-freq trend)")
            axes[panel_i].grid(True, alpha=0.3)
            panel_i += 1

            axes[panel_i].plot(df.index, df["Deviation_Detail"],
                               color=C_RED, linewidth=0.8, alpha=0.9)
            axes[panel_i].fill_between(df.index, df["Deviation_Detail"],
                                       alpha=0.2, color=C_RED)
            axes[panel_i].set_ylabel("D1 (Detail)", fontsize=9)
            axes[panel_i].set_title("Detail Coefficients (High-freq noise)")
            axes[panel_i].grid(True, alpha=0.3)

        else:
            # Chỉ có raw Deviation — 1 panel
            col = "Deviation" if has_raw else df.columns[0]
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(df.index, df[col], color=C_BLUE, linewidth=0.9, alpha=0.9)
            ax.fill_between(df.index, df[col], alpha=0.15, color=C_BLUE)
            ax.axhline(y=0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_ylabel(col, fontsize=11)
            ax.set_title(
                f"{ticker} — Wavelet Decomposition Visualization (Fig. 5)",
                fontsize=13)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
    return fig


# =============================================================================
# FIG. 6 — Wavelet & Scaling Functions (static, pywt db4)
# =============================================================================

def fig6_wavelet_functions() -> plt.Figure:
    """
    Tái hiện Fig. 6: db4 wavelet function (psi) và scaling function (phi).

    Dùng pywt.Wavelet('db4').wavefun(level=10) để tính xấp xỉ liên tục.
    Left panel:  Wavelet function ψ (Mother wavelet)
    Right panel: Scaling function φ (Father wavelet)

    Returns:
        matplotlib.Figure
    """
    try:
        import pywt
    except ImportError:
        logger.warning("pywt không được cài đặt — trả về placeholder figure")
        with plt.rc_context(DARK_RC):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "pip install pywavelets",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color=C_RED)
            ax.axis("off")
        return fig

    # pywt.Wavelet('db4').wavefun(level) → (phi, psi, x)
    # phi = scaling function, psi = wavelet function
    wavelet = pywt.Wavelet("db4")
    phi, psi, x = wavelet.wavefun(level=10)

    with plt.rc_context(DARK_RC):
        fig, (ax_psi, ax_phi) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Wavelet function (ψ — Mother wavelet)
        ax_psi.plot(x, psi, color=C_BLUE, linewidth=2.0)
        ax_psi.fill_between(x, psi, alpha=0.15, color=C_BLUE)
        ax_psi.axhline(y=0, color=C_EDGE, linewidth=0.8, linestyle="--")
        ax_psi.set_title("Wavelet Function ψ (db4)", fontsize=12, fontweight="bold")
        ax_psi.set_xlabel("Time", fontsize=10)
        ax_psi.set_ylabel("Amplitude", fontsize=10)
        ax_psi.grid(True, alpha=0.35)
        ax_psi.set_xlim(x.min(), x.max())

        # Right: Scaling function (φ — Father wavelet)
        ax_phi.plot(x, phi, color=C_GREEN, linewidth=2.0)
        ax_phi.fill_between(x, phi, alpha=0.15, color=C_GREEN)
        ax_phi.axhline(y=0, color=C_EDGE, linewidth=0.8, linestyle="--")
        ax_phi.set_title("Scaling Function φ (db4)", fontsize=12, fontweight="bold")
        ax_phi.set_xlabel("Time", fontsize=10)
        ax_phi.set_ylabel("Amplitude", fontsize=10)
        ax_phi.grid(True, alpha=0.35)
        ax_phi.set_xlim(x.min(), x.max())

        fig.suptitle("db4 Wavelet & Scaling Functions — Fig. 6",
                     fontsize=13, y=1.01)
        plt.tight_layout()
    return fig


# =============================================================================
# FIG. 7 — Approximation Coefficients (multi-panel time series)
# =============================================================================

def fig7_approx_coefficients(df_wavelet: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Tái hiện Fig. 7: Multi-panel line plots của Approximation Coefficients (A1).

    VIC ↔ Tesla trong bài báo gốc.
    Plot tất cả *_Approx columns có trong df_wavelet.

    Args:
        df_wavelet: DataFrame với wavelet features.
        ticker:     Mã cổ phiếu (VIC).

    Returns:
        matplotlib.Figure
    """
    approx_cols = [c for c in df_wavelet.columns if c.endswith("_Approx")]
    return _plot_wavelet_coefficients(df_wavelet, approx_cols, ticker,
                                      coeff_type="Approximation (A1)", fig_num=7)


# =============================================================================
# FIG. 8 — Detail Coefficients (multi-panel time series)
# =============================================================================

def fig8_detail_coefficients(df_wavelet: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Tái hiện Fig. 8: Multi-panel line plots của Detail Coefficients (D1).

    VCB ↔ Apple trong bài báo gốc.
    Plot tất cả *_Detail columns có trong df_wavelet.

    Args:
        df_wavelet: DataFrame với wavelet features.
        ticker:     Mã cổ phiếu (VCB).

    Returns:
        matplotlib.Figure
    """
    detail_cols = [c for c in df_wavelet.columns if c.endswith("_Detail")]
    return _plot_wavelet_coefficients(df_wavelet, detail_cols, ticker,
                                      coeff_type="Detail (D1)", fig_num=8)


def _plot_wavelet_coefficients(
    df: pd.DataFrame,
    cols: list[str],
    ticker: str,
    coeff_type: str,
    fig_num: int,
) -> plt.Figure:
    """
    Helper chung cho fig7/fig8: vẽ multi-panel wavelet coefficient time series.
    """
    if not cols:
        with plt.rc_context(DARK_RC):
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.text(0.5, 0.5, f"Không tìm thấy {coeff_type} features",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color=C_FG)
            ax.axis("off")
        return fig

    n = len(cols)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(DARK_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3 * nrows),
                                 sharex=True)
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(cols):
            ax = axes_flat[i]
            data = df[col].dropna()

            # Lấy tên gốc (bỏ _Approx / _Detail suffix)
            base_name = col.replace("_Approx", "").replace("_Detail", "")
            color = MODEL_PALETTE[i % len(MODEL_PALETTE)]

            ax.plot(data.index, data.values,
                    color=color, linewidth=0.9, alpha=0.9)
            ax.fill_between(data.index, data.values,
                            alpha=0.12, color=color)
            ax.set_ylabel(base_name, fontsize=9)
            ax.set_title(f"{base_name} — {coeff_type}", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)

        # Ẩn axes thừa
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"{ticker} — {coeff_type} Coefficients (Fig. {fig_num})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
    return fig


# =============================================================================
# FIG. 9 — Level-1 Decomposition Diagram (static)
# =============================================================================

def fig9_level1_decomposition() -> plt.Figure:
    """
    Tái hiện Fig. 9: Sơ đồ SWT level-1 decomposition.

    s(t) → Low-pass  filter → A1 (Approximation — low-frequency trend)
    s(t) → High-pass filter → D1 (Detail — high-frequency noise)

    SWT (Stationary Wavelet Transform): không downsample → len(A1) = len(D1) = len(s)

    Returns:
        matplotlib.Figure
    """
    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 5)
        ax.axis("off")

        # ── Input signal ─────────────────────────────────────────────────
        _draw_box(ax, 1.5, 2.5, 2.0, 1.0, "Signal s(t)\n(OHLCV feature)", C_BLUE, fontsize=10)

        # ── Filter boxes ─────────────────────────────────────────────────
        _draw_box(ax, 5.0, 3.8, 2.5, 0.9, "Low-pass Filter\n(db4, level-1)", C_GREEN, fontsize=9)
        _draw_box(ax, 5.0, 1.2, 2.5, 0.9, "High-pass Filter\n(db4, level-1)", C_RED, fontsize=9)

        # ── Output signals ───────────────────────────────────────────────
        _draw_box(ax, 9.5, 3.8, 2.5, 0.9,
                  "A1 — Approximation\n(Trend, low-freq)", C_GREEN, fontsize=9)
        _draw_box(ax, 9.5, 1.2, 2.5, 0.9,
                  "D1 — Detail\n(Noise, high-freq)", C_RED, fontsize=9)

        # ── Arrows ───────────────────────────────────────────────────────
        # Input → bifurcate
        _arrow(ax, 2.5, 2.5, 3.5, 2.5)               # right from input
        _arrow(ax, 3.5, 2.5, 3.5, 3.8)               # up to low-pass
        _arrow(ax, 3.5, 3.8, 3.75, 3.8)              # → low-pass filter
        _arrow(ax, 3.5, 2.5, 3.5, 1.2)               # down to high-pass
        _arrow(ax, 3.5, 1.2, 3.75, 1.2)              # → high-pass filter
        _arrow(ax, 6.25, 3.8, 8.25, 3.8)             # low-pass → A1
        _arrow(ax, 6.25, 1.2, 8.25, 1.2)             # high-pass → D1

        # SWT note: no downsampling
        ax.text(7.25, 4.3, "len(A1) = len(s(t))", fontsize=8,
                ha="center", color=C_GREEN, style="italic")
        ax.text(7.25, 0.65, "len(D1) = len(s(t))", fontsize=8,
                ha="center", color=C_RED, style="italic")
        ax.text(3.5, 2.5, "×2\n(both)", fontsize=8,
                ha="center", va="center", color=C_FG, alpha=0.6)

        # SWT vs DWT note
        ax.text(6.0, 0.1,
                "SWT (Stationary WT): no downsampling → shift-invariant, length-preserving",
                ha="center", fontsize=9, color=C_TEAL, style="italic")

        ax.set_title(
            "Level-1 SWT Decomposition — Li et al. (EAAI 2026), Fig. 9",
            fontsize=13, color=C_FG, pad=14,
        )
        plt.tight_layout()
    return fig


# =============================================================================
# FIG. 10 — Correlation Matrix Heatmap
# =============================================================================

def fig10_correlation_matrix(df_wavelet: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Tái hiện Fig. 10: Correlation matrix heatmap sau wavelet transformation.

    Hiển thị Pearson correlations giữa tất cả feature columns (bao gồm cả Close
    nếu có trong df_wavelet, và cả nowave features nếu truyền raw df).
    Annotate giá trị correlation.

    Args:
        df_wavelet: DataFrame với feature columns (wavelet hoặc raw).
        ticker:     Mã cổ phiếu.

    Returns:
        matplotlib.Figure
    """
    # Chỉ giữ numeric columns
    df_num = df_wavelet.select_dtypes(include=[np.number])
    if df_num.empty:
        with plt.rc_context(DARK_RC):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.text(0.5, 0.5, "Không có numeric columns",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        return fig

    corr = df_num.corr()
    n = len(corr)
    fig_size = max(8, n * 0.9)

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        # Tạo custom colormap: blue (negative) → white (0) → red (positive)
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlBu_r   # Red-Yellow-Blue reversed: dark blue=negative, dark red=positive

        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation", fontsize=10, color=C_FG)
        cbar.ax.yaxis.set_tick_params(color=C_FG)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_FG)

        # Tick labels
        labels = list(corr.columns)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate correlation values
        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                text_color = "white" if abs(val) > 0.7 else C_FG
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

        ax.set_title(
            f"{ticker} — Feature Correlation Matrix (Fig. 10)\n"
            "High corr (|r| > 0.95) features are removed during selection",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
    return fig


# =============================================================================
# FIG. 11 — MSE Comparison Bar Chart (replica Fig. 11 bài báo)
# =============================================================================

def fig11_mse_comparison(ticker: str, currency: str) -> plt.Figure:
    """
    Tái hiện Fig. 11: Bar chart so sánh MSE của 5 models × Before/After Wavelet.

    Đọc kết quả từ evaluation_service.load_all_results().

    Args:
        ticker:   "VCB" hoặc "VIC".
        currency: "VND" hoặc "USD".

    Returns:
        matplotlib.Figure

    Raises:
        ValueError: Nếu không có kết quả experiments.
    """
    from app.services.evaluation_service import load_all_results, create_comparison_table

    df_results = load_all_results()
    if df_results.empty:
        raise ValueError("Không có experiment results. Chạy run_experiments.py trước.")

    tbl = create_comparison_table(df_results, ticker=ticker,
                                  currency=currency, task="regression")
    if tbl.empty:
        raise ValueError(f"Không có data cho {ticker}/{currency} regression.")

    models = list(tbl.index)
    x      = np.arange(len(models))
    width  = 0.35

    # Lấy MSE values (mean across folds)
    try:
        before_mse = tbl[("Before Wavelet", "MSE")].fillna(0).tolist()
        after_mse  = tbl[("After Wavelet",  "MSE")].fillna(0).tolist()
    except KeyError:
        # MultiIndex column có thể không đúng nếu data partial
        before_mse = [0] * len(models)
        after_mse  = [0] * len(models)

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(11, 6))

        bars1 = ax.bar(x - width / 2, before_mse, width,
                       label="Before Wavelet", color="#4472C4", alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, after_mse, width,
                       label="After Wavelet", color=C_GREEN, alpha=0.85,
                       edgecolor="white", linewidth=0.5)

        # Auto log scale nếu giá trị chênh lệch lớn
        vals = [v for v in before_mse + after_mse if v > 0]
        if vals and max(vals) / (min(vals) + 1e-12) > 100:
            ax.set_yscale("log")
            ax.set_ylabel("MSE (log scale)", fontsize=11)
        else:
            ax.set_ylabel("MSE", fontsize=11)

        # Annotate bars
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3g}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=C_FG)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3g}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=C_FG)

        ax.set_title(
            f"MSE Comparison — {ticker} ({currency})\n"
            "Before vs After Wavelet Transformation (Fig. 11)",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.35)
        plt.tight_layout()
    return fig


# =============================================================================
# EXTENDED: Walk-Forward Stability Chart
# =============================================================================

def ext_walkforward_stability(
    ticker   : str,
    currency : str,
    task     : str = "regression",
    metric   : str = "MSE",
    wavelet  : bool = True,
) -> plt.Figure:
    """
    Extended Fig: Walk-forward stability — metric per fold per model.

    Cho thấy consistency của mỗi model qua 3 folds.
    Regression: MSE, MAE, MAPE, R2. Classification: Accuracy, F1, AUC_ROC.

    Args:
        ticker, currency: Filter kết quả.
        task:    "regression" hoặc "classification".
        metric:  Tên metric cần hiện (MSE, F1, Accuracy...).
        wavelet: True = After Wavelet, False = Before Wavelet.

    Returns:
        matplotlib.Figure
    """
    from app.services.evaluation_service import load_all_results
    from app.config import MODELS, FOLDS

    df_results = load_all_results()
    if df_results.empty:
        raise ValueError("Không có kết quả experiments.")

    cond_str = "wavelet" if wavelet else "nowave"
    df_fil = df_results[
        (df_results["ticker"]       == ticker)
        & (df_results["currency"]   == currency)
        & (df_results["task"]       == task)
        & (df_results["use_wavelet"] == wavelet)
    ].copy()

    if df_fil.empty or metric not in df_fil.columns:
        raise ValueError(f"Không có data cho {ticker}/{currency}/{task}/{metric}")

    folds_avail = sorted(df_fil["fold"].unique())

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, model_name in enumerate(MODELS):
            df_m = df_fil[df_fil["model"] == model_name].sort_values("fold")
            if df_m.empty:
                continue
            folds  = df_m["fold"].tolist()
            values = df_m[metric].tolist()
            color  = MODEL_PALETTE[i % len(MODEL_PALETTE)]
            ax.plot(folds, values, marker="o", label=model_name,
                    color=color, linewidth=2.0, markersize=7)

        ax.set_title(
            f"Walk-Forward Stability — {ticker} ({currency})\n"
            f"{metric} per Fold | {cond_str} | {task}",
            fontsize=13,
        )
        ax.set_xlabel("Fold", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks(sorted(df_fil["fold"].unique()))
        ax.set_xticklabels([f"Fold {f}" for f in sorted(df_fil["fold"].unique())])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)
        plt.tight_layout()
    return fig


# =============================================================================
# EXTENDED: Multi-model ROC Curves
# =============================================================================

def ext_roc_curves(
    ticker   : str,
    currency : str,
    fold_idx : int = 3,
    wavelet  : bool = True,
) -> plt.Figure:
    """
    Extended Fig: Overlaid ROC curves cho 5 models trên cùng test fold.

    Args:
        ticker, currency: Xác định dataset.
        fold_idx: Fold number (1, 2, 3).
        wavelet:  True = wavelet condition.

    Returns:
        matplotlib.Figure
    """
    import pickle
    from pathlib import Path
    from app.config import MODELS, PATHS
    from sklearn.metrics import roc_curve, auc

    cond = "wavelet" if wavelet else "nowave"
    exp_dir = Path(PATHS["experiments"])

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(8, 7))

        any_plotted = False
        for i, model_name in enumerate(MODELS):
            exp_id   = f"{ticker}_{currency}_{cond}_{model_name}_classification"
            npz_path = exp_dir / exp_id / f"fold_{fold_idx}" / "predictions.npz"
            if not npz_path.exists():
                continue

            try:
                data    = np.load(str(npz_path), allow_pickle=False)
                y_true  = data["y_true"].flatten()
                y_prob  = data["y_prob"].flatten() if "y_prob" in data else None

                if y_prob is None or len(np.unique(y_true)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc     = auc(fpr, tpr)
                color       = MODEL_PALETTE[i % len(MODEL_PALETTE)]

                ax.plot(fpr, tpr, color=color, linewidth=2.0,
                        label=f"{model_name} (AUC = {roc_auc:.3f})")
                any_plotted = True
            except Exception as exc:
                logger.warning("ROC curve error for %s: %s", exp_id, exc)

        if not any_plotted:
            ax.text(0.5, 0.5, "Không có classification predictions",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color=C_FG)
        else:
            # Diagonal (random classifier)
            ax.plot([0, 1], [0, 1], color=C_EDGE, linewidth=1.5,
                    linestyle="--", alpha=0.7, label="Random (AUC=0.5)")

        ax.set_title(
            f"ROC Curves — {ticker} ({currency}) | {cond} | Fold {fold_idx}",
            fontsize=13,
        )
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.35)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        plt.tight_layout()
    return fig


# =============================================================================
# TASK 9.2 — EXTENDED VISUALIZATIONS (Functions 12–19)
# =============================================================================
# Helper chung: lưu file nếu save_path được cung cấp
# =============================================================================

def _save_fig(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Lưu figure ra disk nếu save_path không None."""
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Figure saved → %s", save_path)


# =============================================================================
# 12. fig_predicted_vs_actual — Line chart Actual vs Predicted
# =============================================================================

def fig_predicted_vs_actual(
    exp_id   : str,
    fold_idx : int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line chart so sánh giá đóng cửa Actual và Predicted trên test set.

    Load predictions từ experiments/{exp_id}/fold_{fold_idx}/predictions.npz.
    X-axis: sequential test day index.
    Y-axis: Close price với đơn vị VND hoặc USD (suy ra từ exp_id).

    Args:
        exp_id:    Regression experiment ID (e.g. "VCB_VND_wavelet_BiLSTM_regression").
        fold_idx:  Fold số (1, 2, 3).
        save_path: Nếu không None, lưu PNG ra đường dẫn này.

    Returns:
        matplotlib.Figure

    Raises:
        FileNotFoundError: predictions.npz không tồn tại.
        ValueError:        exp_id không phải regression.
    """
    from pathlib import Path
    from app.config import PATHS

    if "classification" in exp_id:
        raise ValueError(f"exp_id '{exp_id}' là classification, cần regression.")

    npz_path = Path(PATHS["experiments"]) / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"predictions.npz không tồn tại: {npz_path}")

    raw      = np.load(str(npz_path), allow_pickle=False)
    y_true   = raw["y_true"].flatten()
    y_pred   = raw["y_pred"].flatten()
    n        = len(y_true)

    # Suy ra currency label từ exp_id
    currency_label = "USD" if "_USD_" in exp_id else "VND"
    # Lấy tên model để hiển thị
    parts      = exp_id.split("_")
    model_name = parts[3] if len(parts) > 3 else exp_id

    # Tính error bands
    errors   = y_pred - y_true
    mae_val  = float(np.mean(np.abs(errors)))
    rmse_val = float(np.sqrt(np.mean(errors ** 2)))

    with plt.rc_context(DARK_RC):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})

        # ── Panel 1: Actual vs Predicted ─────────────────────────────────
        x = np.arange(n)
        axes[0].plot(x, y_true, label="Actual",
                     color=C_BLUE, linewidth=1.3, alpha=0.95, zorder=3)
        axes[0].plot(x, y_pred, label="Predicted",
                     color=C_RED, linewidth=1.1, alpha=0.85,
                     linestyle="--", zorder=4)

        # Shade error region
        axes[0].fill_between(x, y_true, y_pred,
                             alpha=0.08, color=C_ORANGE, label="Error region")

        axes[0].set_title(
            f"Predicted vs Actual Close Price — {model_name} | Fold {fold_idx}",
            fontsize=13, fontweight="bold",
        )
        axes[0].set_ylabel(f"Close Price ({currency_label})", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Stats annotation
        axes[0].text(0.02, 0.97,
                     f"MAE: {mae_val:,.2f} {currency_label}\nRMSE: {rmse_val:,.2f} {currency_label}",
                     transform=axes[0].transAxes,
                     va="top", ha="left", fontsize=9,
                     color=C_FG, alpha=0.85,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a3a",
                               edgecolor=C_EDGE, alpha=0.9))

        # ── Panel 2: Residuals ────────────────────────────────────────────
        axes[1].bar(x, errors, color=np.where(errors >= 0, C_GREEN, C_RED),
                    alpha=0.7, width=0.8)
        axes[1].axhline(y=0, color=C_FG, linewidth=0.8, linestyle="--")
        axes[1].set_ylabel("Residual", fontsize=10)
        axes[1].set_xlabel("Test Day Index", fontsize=11)
        axes[1].set_title("Residuals (Predicted − Actual)", fontsize=10)
        axes[1].grid(True, alpha=0.25)

        plt.tight_layout()

    _save_fig(fig, save_path)
    return fig


# =============================================================================
# 13. fig_loss_curves — Train/Val loss per epoch với best epoch marker
# =============================================================================

def fig_loss_curves(
    exp_id   : str,
    fold_idx : int = 1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line chart Train loss và Val loss per epoch.

    Load từ experiments/{exp_id}/fold_{fold_idx}/metrics.json.
    Vertical dashed line tại best epoch (early stopping).
    Annotation: best val loss value.

    Args:
        exp_id:    Experiment ID (regression hoặc classification).
        fold_idx:  Fold số (1, 2, 3).
        save_path: Path để lưu PNG (optional).

    Returns:
        matplotlib.Figure
    """
    import json
    from pathlib import Path
    from app.config import PATHS

    json_path = Path(PATHS["experiments"]) / exp_id / f"fold_{fold_idx}" / "metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(f"metrics.json không tồn tại: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    history      = data.get("train_history", {})
    train_losses = history.get("train_losses", [])
    val_losses   = history.get("val_losses", [])
    best_epoch   = history.get("best_epoch", None)
    best_val     = history.get("best_val_loss", None)

    if not train_losses:
        raise ValueError(f"train_losses rỗng trong metrics.json ({exp_id}/fold_{fold_idx})")

    n_epochs = len(train_losses)
    epochs   = list(range(1, n_epochs + 1))

    # Model name từ exp_id
    parts      = exp_id.split("_")
    model_name = parts[3] if len(parts) > 3 else exp_id

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(epochs, train_losses, label="Train Loss",
                color=C_BLUE, linewidth=1.8, alpha=0.9)
        ax.plot(epochs, val_losses, label="Val Loss",
                color=C_RED, linewidth=1.8, linestyle="--", alpha=0.9)

        # Best epoch vertical line
        if best_epoch:
            ax.axvline(x=best_epoch, color=C_GREEN, linestyle=":",
                       linewidth=2.2, label=f"Best Epoch = {best_epoch}",
                       zorder=5)
            # Annotate best val loss
            if best_val is not None:
                ax.annotate(
                    f"Best val: {best_val:.5f}",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + n_epochs * 0.04, best_val),
                    fontsize=8.5, color=C_GREEN,
                    arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.0),
                )

        # Shade overfitting region (val > train)
        if len(train_losses) == len(val_losses):
            t_arr = np.array(train_losses)
            v_arr = np.array(val_losses)
            overfit_mask = v_arr > t_arr
            if overfit_mask.any():
                ax.fill_between(epochs, t_arr, v_arr,
                                where=overfit_mask,
                                alpha=0.06, color=C_RED, label="Overfit region")

        ax.set_title(
            f"Loss Curves — {model_name} | Fold {fold_idx} | {n_epochs} epochs",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    _save_fig(fig, save_path)
    return fig


# =============================================================================
# 14. fig_classification_table — Accuracy/F1/AUC per model per fold
# =============================================================================

def fig_classification_table(
    ticker   : str,
    currency : str,
    wavelet  : bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Matplotlib figure chứa bảng Classification metrics.

    Hiển thị Accuracy / F1 / AUC_ROC cho 5 models × 3 folds + Mean.
    Best model mỗi metric được highlight màu vàng.

    Args:
        ticker, currency: Filter kết quả.
        wavelet:   True = After Wavelet, False = Before Wavelet.
        save_path: Path lưu PNG (optional).

    Returns:
        matplotlib.Figure
    """
    from app.services.evaluation_service import load_all_results
    from app.config import MODELS

    df_all = load_all_results()
    if df_all.empty:
        raise ValueError("Chưa có experiment results.")

    df_fil = df_all[
        (df_all["ticker"]       == ticker)
        & (df_all["currency"]   == currency)
        & (df_all["use_wavelet"] == wavelet)
        & (df_all["task"]       == "classification")
    ].copy()

    if df_fil.empty:
        raise ValueError(f"Không có classification data: {ticker}/{currency}/wavelet={wavelet}")

    # ── Tạo bảng: rows=models, columns=Fold1/Fold2/Fold3/Mean × 3 metrics
    metrics_show = ["Accuracy", "F1", "AUC_ROC"]
    folds_avail  = sorted(df_fil["fold"].unique())   # [1, 2, 3]

    # Multi-level column: (metric, fold_label)
    col_labels = []
    for m in metrics_show:
        for f in folds_avail:
            col_labels.append(f"{m}\nFold {f}")
        col_labels.append(f"{m}\nMean")

    cell_text  = []
    cell_colors = []
    model_means = {m: {} for m in metrics_show}  # model → mean value per metric

    for model_name in MODELS:
        df_m = df_fil[df_fil["model"] == model_name]
        row_vals   = []
        row_colors = []
        for metric in metrics_show:
            fold_vals = []
            for f in folds_avail:
                val = df_m[df_m["fold"] == f][metric].values
                v   = float(val[0]) if len(val) > 0 and not np.isnan(val[0]) else np.nan
                fold_vals.append(v)
                row_vals.append(f"{v:.4f}" if not np.isnan(v) else "—")
                row_colors.append("#1a1a3a")  # default dark
            mean_v = float(np.nanmean(fold_vals)) if fold_vals else np.nan
            model_means[metric][model_name] = mean_v
            row_vals.append(f"{mean_v:.4f}" if not np.isnan(mean_v) else "—")
            row_colors.append("#1a2a3a")  # mean column slightly different
        cell_text.append(row_vals)
        cell_colors.append(row_colors)

    # Highlight best model per metric (Mean column)
    n_metrics  = len(metrics_show)
    n_folds    = len(folds_avail)
    cols_per_m = n_folds + 1   # folds + mean

    for mi, metric in enumerate(metrics_show):
        mean_col_idx = mi * cols_per_m + n_folds  # index of Mean column
        best_model   = max(model_means[metric], key=lambda m: model_means[metric].get(m, -1))
        best_row_idx = MODELS.index(best_model) if best_model in MODELS else -1
        if best_row_idx >= 0:
            cell_colors[best_row_idx][mean_col_idx] = "#2a4a00"  # highlight green

    # ── Render matplotlib table ────────────────────────────────────────
    with plt.rc_context(DARK_RC):
        fig_h = max(4, len(MODELS) * 0.65 + 2.5)
        fig_w = max(14, len(col_labels) * 1.1 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        # Header row
        cond_str = "After Wavelet" if wavelet else "Before Wavelet"
        ax.set_title(
            f"Classification Metrics — {ticker} ({currency}) | {cond_str}\n"
            "Highlighted (green) = Best Mean per Metric",
            fontsize=12, fontweight="bold", y=1.02,
        )

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=MODELS,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.8)

        # Apply colors
        for r_i, model_name in enumerate(MODELS):
            # Row label (leftmost)
            tbl[r_i + 1, -1].set_facecolor("#0f1a2a")
            tbl[r_i + 1, -1].set_text_props(color=C_FG, fontweight="bold")
            for c_i in range(len(col_labels)):
                cell = tbl[r_i + 1, c_i]
                cell.set_facecolor(cell_colors[r_i][c_i])
                cell.set_text_props(color=C_FG)
                cell.set_edgecolor(C_EDGE)

        # Column header styling
        for c_i in range(len(col_labels)):
            hdr = tbl[0, c_i]
            hdr.set_facecolor("#0a1a3a")
            hdr.set_text_props(color=C_BLUE, fontweight="bold", fontsize=8)
            hdr.set_edgecolor(C_EDGE)

        plt.tight_layout()

    _save_fig(fig, save_path)
    return fig


# =============================================================================
# 15. fig_confusion_matrix — 2×2 heatmap với annotations
# =============================================================================

def fig_confusion_matrix(
    exp_id   : str,
    fold_idx : int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2×2 Confusion Matrix heatmap với count và percentage annotations.

    Load từ experiments/{exp_id}/fold_{fold_idx}/predictions.npz.
    Chỉ hợp lệ cho classification experiments.

    Args:
        exp_id:    Classification experiment ID.
        fold_idx:  Fold số (1, 2, 3).
        save_path: Path lưu PNG (optional).

    Returns:
        matplotlib.Figure

    Raises:
        ValueError: exp_id không phải classification.
    """
    from pathlib import Path
    from sklearn.metrics import confusion_matrix
    from app.config import PATHS

    if "classification" not in exp_id:
        raise ValueError(f"exp_id '{exp_id}' phải chứa 'classification'.")

    npz_path = Path(PATHS["experiments"]) / exp_id / f"fold_{fold_idx}" / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"predictions.npz không tồn tại: {npz_path}")

    raw    = np.load(str(npz_path), allow_pickle=False)
    y_true = raw["y_true"].flatten().astype(int)
    y_pred = raw["y_pred"].flatten().astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    parts      = exp_id.split("_")
    model_name = parts[3] if len(parts) > 3 else exp_id
    total      = cm.sum()

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(6, 5.5))

        im  = ax.imshow(cm, interpolation="nearest",
                        cmap=plt.cm.Blues, vmin=0, vmax=cm.max() * 1.2)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=C_FG)
        cbar.set_label("Count", color=C_FG, fontsize=9)

        labels = ["DOWN (0)", "UP (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(
            f"Confusion Matrix — {model_name} | Fold {fold_idx}",
            fontsize=13, fontweight="bold",
        )

        # Annotate: count + percentage
        thresh = cm.max() / 2.0
        cell_names = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val     = cm[i, j]
                pct     = 100.0 * val / total if total > 0 else 0.0
                txt_col = "white" if val > thresh else C_FG
                ax.text(j, i,
                        f"{cell_names[i][j]}\n{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=11, color=txt_col, fontweight="bold")

        plt.tight_layout()

    _save_fig(fig, save_path)
    return fig


# =============================================================================
# 16. fig_roc_curves — ROC curves overlay cho 5 models
# =============================================================================

def fig_roc_curves(
    ticker   : str,
    currency : str,
    fold_idx : int = 3,
    wavelet  : bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay ROC curves của 5 models + diagonal reference line.

    AUC per model hiển thị trong legend.
    Tổng hợp toàn bộ classification predictions.

    Args:
        ticker, currency: Xác định dataset.
        fold_idx:   Fold số (1, 2, 3).
        wavelet:    True = wavelet condition.
        save_path:  Path lưu PNG (optional).

    Returns:
        matplotlib.Figure
    """
    # Delegate to existing ext_roc_curves + save_path support
    fig = ext_roc_curves(ticker, currency, fold_idx, wavelet)
    _save_fig(fig, save_path)
    return fig


# =============================================================================
# 17. fig_cumulative_return — Multi-line cumulative returns + Sharpe/MaxDD
# =============================================================================

def fig_cumulative_return(
    ticker   : str,
    currency : str,
    fold_idx : int = 3,
    wavelet  : bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-line chart: Cumulative returns của 5 models + Buy&Hold baseline.

    X-axis: date (từ df_test.index).
    Y-axis: Cumulative return (%).
    Annotation: Sharpe Ratio và Max Drawdown per model.

    Args:
        ticker, currency: Xác định dataset.
        fold_idx:   Fold số (1, 2, 3).
        wavelet:    True = wavelet condition.
        save_path:  Path lưu PNG (optional).

    Returns:
        matplotlib.Figure
    """
    import pickle
    from pathlib import Path
    import matplotlib.dates as mdates
    from app.config import MODELS, PATHS, FOLDS
    from app.services.trading_service import simulate_trading, compute_trading_metrics

    cond     = "wavelet" if wavelet else "nowave"
    fold_def = next((f for f in FOLDS if f["fold_id"] == fold_idx), None)
    if fold_def is None:
        raise ValueError(f"fold_idx={fold_idx} không hợp lệ.")

    test_start = pd.Timestamp(fold_def["test_start"])
    test_end   = pd.Timestamp(fold_def["test_end"])

    # Load processed pkl (dict format)
    pkl_path = Path(PATHS["processed"]) / f"{ticker}_{currency}_{cond}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Processed pkl không tồn tại: {pkl_path}")
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    df_full   = pkl_data["df"] if isinstance(pkl_data, dict) else pkl_data
    df_test   = df_full[(df_full.index >= test_start) & (df_full.index <= test_end)]
    close_arr = df_test["Close"].values.astype(float)
    T         = len(close_arr)

    if T < 5:
        raise ValueError("Không đủ dữ liệu giá trong test period.")

    with plt.rc_context(DARK_RC):
        # ── Main chart + stats panel ───────────────────────────────────
        fig = plt.figure(figsize=(15, 8))
        gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_main  = fig.add_subplot(gs[0])
        ax_stats = fig.add_subplot(gs[1])
        ax_stats.axis("off")

        stats_rows   = []
        buyhold_done = False
        exp_dir      = Path(PATHS["experiments"])

        for i, model_name in enumerate(MODELS):
            exp_id   = f"{ticker}_{currency}_{cond}_{model_name}_classification"
            npz_path = exp_dir / exp_id / f"fold_{fold_idx}" / "predictions.npz"
            if not npz_path.exists():
                continue

            try:
                raw    = np.load(str(npz_path), allow_pickle=False)
                y_pred = raw["y_pred"].flatten().astype(int)
            except Exception as exc:
                logger.warning("Lỗi load %s: %s", npz_path, exc)
                continue

            n_pred = len(y_pred)
            if n_pred >= T:
                continue

            seq_len    = T - n_pred
            act_prices = close_arr[seq_len - 1:]           # shape (n_pred + 1,)
            dates_idx  = df_test.index[seq_len - 1: seq_len - 1 + n_pred]

            trade_df  = simulate_trading(y_pred, act_prices, dates=dates_idx.values)
            trade_mtr = compute_trading_metrics(trade_df)

            color = MODEL_PALETTE[i % len(MODEL_PALETTE)]

            # Date-based X-axis
            plot_dates  = trade_df.index if hasattr(trade_df.index, 'year') else dates_idx
            strat_pct   = (trade_df["Strategy_Cumulative"] - 1) * 100
            ax_main.plot(plot_dates, strat_pct,
                         label=model_name, color=color,
                         linewidth=1.8, alpha=0.9)

            # Buy & Hold — plot một lần
            if not buyhold_done:
                bh_pct = (trade_df["BuyHold_Cumulative"] - 1) * 100
                ax_main.plot(plot_dates, bh_pct,
                             label="Buy & Hold", color="#78909c",
                             linewidth=2.2, linestyle="--", alpha=0.85)
                buyhold_done = True

            # Collect stats
            stats_rows.append({
                "Model"          : model_name,
                "Cum. Return"    : f"{trade_mtr.get('Cumulative_Return', 0)*100:.1f}%",
                "B&H Return"     : f"{trade_mtr.get('BuyHold_Return', 0)*100:.1f}%",
                "Sharpe"         : f"{trade_mtr.get('Sharpe_Ratio', 0):.3f}",
                "Max DD"         : f"{trade_mtr.get('Max_Drawdown', 0)*100:.1f}%",
                "Win Rate"       : f"{trade_mtr.get('Win_Rate', 0)*100:.1f}%",
            })

        ax_main.axhline(y=0, color="#444466", linewidth=0.8, linestyle=":")
        ax_main.set_title(
            f"Cumulative Returns — {ticker} ({currency}) | {cond} | Fold {fold_idx}",
            fontsize=13, fontweight="bold",
        )
        ax_main.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax_main.legend(fontsize=10, loc="upper left")
        ax_main.grid(True, alpha=0.3)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_main.xaxis.get_ticklabels(), rotation=30, ha="right", fontsize=8)

        # ── Stats table panel ─────────────────────────────────────────
        if stats_rows:
            cols  = list(stats_rows[0].keys())
            cells = [[row[c] for c in cols] for row in stats_rows]
            tbl   = ax_stats.table(
                cellText=cells,
                colLabels=cols,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1, 1.6)

            # Style header
            for c_i, col in enumerate(cols):
                hdr = tbl[0, c_i]
                hdr.set_facecolor("#0a1a3a")
                hdr.set_text_props(color=C_BLUE, fontweight="bold")
                hdr.set_edgecolor(C_EDGE)
            # Style cells
            for r_i in range(len(stats_rows)):
                for c_i in range(len(cols)):
                    cell = tbl[r_i + 1, c_i]
                    cell.set_facecolor("#1a1a3a")
                    cell.set_text_props(color=C_FG)
                    cell.set_edgecolor(C_EDGE)

        plt.tight_layout()

    _save_fig(fig, save_path)
    return fig


# =============================================================================
# 18. fig_walkforward_stability — Bar chart metrics per fold cho 1 exp_id
# =============================================================================

def fig_walkforward_stability(
    exp_id   : str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart so sánh metrics qua 3 folds cho một experiment cụ thể.

    Tự detect task (regression/classification) từ exp_id.
    Hiện tất cả metrics liên quan của task đó.

    Args:
        exp_id:    Experiment ID (regression hoặc classification).
        save_path: Path lưu PNG (optional).

    Returns:
        matplotlib.Figure
    """
    import json
    from pathlib import Path
    from app.config import FOLDS, PATHS

    exp_dir = Path(PATHS["experiments"]) / exp_id

    # Detect task
    is_regression = "regression" in exp_id
    metrics_to_show = (
        ["mse", "mae", "mape", "r2"] if is_regression
        else ["accuracy", "f1", "auc_roc"]
    )
    metric_labels = (
        ["MSE", "MAE", "MAPE (%)", "R²"] if is_regression
        else ["Accuracy", "F1", "AUC-ROC"]
    )

    # Load metrics per fold
    fold_ids = [f["fold_id"] for f in FOLDS]
    fold_data = {}   # fold_id → metrics dict
    for fid in fold_ids:
        json_path = exp_dir / f"fold_{fid}" / "metrics.json"
        if not json_path.exists():
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        fold_data[fid] = data.get("metrics", {})

    if not fold_data:
        raise FileNotFoundError(f"Không tìm thấy metrics.json trong: {exp_dir}")

    folds_avail = sorted(fold_data.keys())
    n_metrics   = len(metrics_to_show)
    n_folds     = len(folds_avail)

    parts      = exp_id.split("_")
    model_name = parts[3] if len(parts) > 3 else exp_id

    with plt.rc_context(DARK_RC):
        fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        fold_colors = [C_BLUE, C_GREEN, C_ORANGE]
        fold_labels = [f"Fold {fid}" for fid in folds_avail]

        x = np.arange(n_folds)
        bar_w = 0.55

        for mi, (metric_key, metric_label) in enumerate(zip(metrics_to_show, metric_labels)):
            ax   = axes[mi]
            vals = [fold_data.get(fid, {}).get(metric_key, np.nan)
                    for fid in folds_avail]

            bars = ax.bar(x, vals, bar_w,
                          color=[fold_colors[i % len(fold_colors)] for i in range(n_folds)],
                          alpha=0.85, edgecolor="white", linewidth=0.7)

            # Annotate bar values
            for bar_obj, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            bar_obj.get_height(),
                            f"{val:.4f}",
                            ha="center", va="bottom", fontsize=8, color=C_FG)

            # Mean ± std line
            valid_vals = [v for v in vals if not np.isnan(v)]
            if valid_vals:
                mean_v = float(np.mean(valid_vals))
                ax.axhline(y=mean_v, color=C_RED, linewidth=1.5,
                           linestyle="--", alpha=0.8,
                           label=f"Mean: {mean_v:.4f}")
                ax.legend(fontsize=8)

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(fold_labels, fontsize=9)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Walk-Forward Stability — {model_name} | {exp_id}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

    _save_fig(fig, save_path)
    return fig


# =============================================================================
# 19. fig_vnd_vs_usd_table — Side-by-side VND vs USD comparison
# =============================================================================

def fig_vnd_vs_usd_table(
    model_name: str,
    save_path : Optional[str] = None,
) -> plt.Figure:
    """
    Matplotlib table figure so sánh VND vs USD side-by-side cho một model.

    Rows: (ticker, wavelet_condition, task)
    Columns: metric × (VND, USD, Δ USD−VND)

    Regression metrics: MSE, MAE, MAPE, R2
    Classification metrics: Accuracy, F1, AUC_ROC

    Args:
        model_name: "DNN", "RNN", "GRU", "LSTM", "BiLSTM".
        save_path:  Path lưu PNG (optional).

    Returns:
        matplotlib.Figure
    """
    from app.services.evaluation_service import load_all_results
    from app.config import TICKERS

    df_all = load_all_results()
    if df_all.empty:
        raise ValueError("Chưa có experiment results.")

    df_m = df_all[df_all["model"] == model_name].copy()
    if df_m.empty:
        raise ValueError(f"Không có data cho model={model_name}")

    # Aggregate mean across folds
    group_cols  = ["ticker", "use_wavelet", "task", "currency"]
    metric_cols = ["MSE", "MAE", "MAPE", "R2",
                   "Accuracy", "F1", "AUC_ROC"]

    df_agg = (
        df_m.groupby(group_cols, observed=True)[metric_cols]
        .mean()
        .reset_index()
    )

    # Pivot: VND vs USD
    df_vnd = df_agg[df_agg["currency"] == "VND"].drop(columns="currency")
    df_usd = df_agg[df_agg["currency"] == "USD"].drop(columns="currency")

    id_cols   = ["ticker", "use_wavelet", "task"]
    df_merged = pd.merge(df_vnd, df_usd, on=id_cols, how="outer",
                         suffixes=("_VND", "_USD"))

    if df_merged.empty:
        raise ValueError(f"Không đủ data để so sánh VND vs USD cho model={model_name}")

    # ── Build table data ────────────────────────────────────────────────
    # Detect columns thực sự có data
    reg_metrics = ["MSE", "MAE", "MAPE", "R2"]
    cls_metrics = ["Accuracy", "F1", "AUC_ROC"]

    def _metric_cols_for_task(task_name):
        return reg_metrics if task_name == "regression" else cls_metrics

    row_labels  = []
    col_labels  = None
    cell_values = []

    for _, row in df_merged.sort_values(["ticker", "task", "use_wavelet"]).iterrows():
        ticker   = row["ticker"]
        wavelet  = row["use_wavelet"]
        task     = row["task"]
        cond_str = "Wave" if wavelet else "No-Wave"
        row_labels.append(f"{ticker}\n{cond_str}\n{task[:3].upper()}")

        metrics_here = _metric_cols_for_task(task)
        if col_labels is None:
            # Build column structure first time
            col_labels = []
            for m in metrics_here:
                col_labels += [f"{m}\nVND", f"{m}\nUSD", f"{m}\nΔ"]

        row_data   = []
        row_colors = []
        for m in metrics_here:
            vnd_v = row.get(f"{m}_VND", np.nan)
            usd_v = row.get(f"{m}_USD", np.nan)
            if pd.notna(vnd_v) and pd.notna(usd_v):
                delta = usd_v - vnd_v
                row_data += [f"{vnd_v:.4f}", f"{usd_v:.4f}", f"{delta:+.4f}"]
                # Color delta: green=improvement, red=worse (depends on metric)
                is_lower_better = m in ["MSE", "MAE", "MAPE"]
                is_usd_better = (delta < 0) if is_lower_better else (delta > 0)
                delta_color = "#1a3a0a" if is_usd_better else "#3a0a0a"
                row_colors += ["#1a1a3a", "#1a1a3a", delta_color]
            else:
                row_data += ["—", "—", "—"]
                row_colors += ["#1a1a3a", "#1a1a3a", "#1a1a3a"]
        cell_values.append((row_data, row_colors))

    if not cell_values:
        raise ValueError("Không có data sau khi merge VND/USD.")

    # Rebuild table để đồng nhất số cột
    if col_labels is None:
        col_labels = ["MSE\nVND", "MSE\nUSD", "MSE\nΔ"]   # fallback

    cell_text   = [row for row, _ in cell_values]
    cell_colors = [colors for _, colors in cell_values]

    # ── Render matplotlib table ────────────────────────────────────────
    n_rows = len(row_labels)
    n_cols = len(col_labels) if col_labels else 3

    fig_w = max(16, n_cols * 1.2 + 2)
    fig_h = max(4, n_rows * 0.8 + 2.5)

    with plt.rc_context(DARK_RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        ax.set_title(
            f"VND vs USD Metric Comparison — {model_name} (mean across 3 folds)\n"
            "Δ = USD − VND | Green Δ = USD outperforms; Red Δ = VND outperforms",
            fontsize=11, fontweight="bold", y=1.03,
        )

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.0)
        tbl.scale(1.0, 1.9)

        # Style cells
        for r_i in range(n_rows):
            for c_i in range(n_cols):
                cell = tbl[r_i + 1, c_i]
                cell.set_facecolor(cell_colors[r_i][c_i])
                cell.set_text_props(color=C_FG)
                cell.set_edgecolor(C_EDGE)
            # Row label
            tbl[r_i + 1, -1].set_facecolor("#0f1a2a")
            tbl[r_i + 1, -1].set_text_props(color=C_FG, fontsize=8)
            tbl[r_i + 1, -1].set_edgecolor(C_EDGE)

        # Header
        for c_i in range(n_cols):
            hdr = tbl[0, c_i]
            hdr.set_facecolor("#0a1a3a")
            hdr.set_text_props(color=C_BLUE, fontweight="bold", fontsize=8)
            hdr.set_edgecolor(C_EDGE)

        plt.tight_layout()

    _save_fig(fig, save_path)
    return fig