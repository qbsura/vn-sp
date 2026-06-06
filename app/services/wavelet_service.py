"""
app/services/wavelet_service.py
=================================
Stationary Wavelet Transform (SWT) cho feature decomposition.

Tham chiếu bài báo Li et al. (2026):
  - Section 3.3.1: DWT db4 level-1, channel-wise, symmetric padding
  - Section 3.3.3: SWT (Stationary Wavelet Transform) — shift-invariant,
    giữ nguyên length, không decimation (khác DWT thông thường)
  - Section 4.5: symmetric padding + trim edge coefficients để giảm
    boundary artifacts

Lưu ý kỹ thuật (verified pywt 1.8.0):
  - pywt.swt KHÔNG nhận `mode` parameter (khác pywt.dwt).
    SWT dùng periodic extension internally → output luôn cùng length với input.
  - pywt.swt yêu cầu input length chia hết cho 2^level.
    Level=1 → cần length chẵn → auto-pad nếu lẻ (wrap mode).
  - pywt.swt returns [(cAn, cDn), ..., (cA1, cD1)] — level-1 là phần tử cuối.
  - db4 filter length = 8 → boundary artifacts ảnh hưởng ~7 samples mỗi đầu.
  - WAVELET_CONFIG["mode"]="periodization" là tham khảo; không truyền vào swt().
"""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from app.config import PATHS, WAVELET_CONFIG

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(PATHS["processed"])

# ── Features cần decompose ────────────────────────────────────────────────────
# Close là target → KHÔNG decompose, giữ nguyên
FEATURES_TO_DECOMPOSE = ["Open", "High", "Low", "Volume", "Deviation"]

# db4 filter length = 8 → boundary artifacts tại 7 samples mỗi đầu (tham khảo)
_DB4_BOUNDARY_EDGE = 7


# =============================================================================
# CORE SWT FUNCTIONS
# =============================================================================

def apply_swt_to_feature(
    series: np.ndarray,
    wavelet: str = "db4",
    level: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Áp dụng Stationary Wavelet Transform (SWT) lên một feature series.

    SWT (section 3.3.3) không downsample → output cùng length với input.
    Dùng periodic extension nội bộ → không cần mode parameter.

    Xử lý odd-length: pywt.swt yêu cầu length chia hết 2^level.
    Level=1 → cần chẵn → pad 1 sample cuối bằng wrap (lặp tròn),
    rồi trim lại sau SWT để khớp length gốc.

    Args:
        series:  1-D array dữ liệu chuỗi thời gian (float).
        wavelet: Tên wavelet — mặc định 'db4' (Daubechies-4).
        level:   Mức decompose — mặc định 1 (bài báo section 3.3.1).

    Returns:
        (approx_coeffs, detail_coeffs): Cả hai có shape == (len(series),).
          - approx_coeffs (cA1): Low-frequency trend component.
          - detail_coeffs (cD1): High-frequency detail / volatility component.

    Raises:
        ValueError: Nếu series rỗng hoặc level < 1.
    """
    data = np.asarray(series, dtype=np.float64).copy()
    n_orig = len(data)

    if n_orig == 0:
        raise ValueError("apply_swt_to_feature: series rỗng.")
    if level < 1:
        raise ValueError(f"apply_swt_to_feature: level={level} phải >= 1.")

    # ── Pad nếu length không chia hết cho 2^level ─────────────────────────────
    # Level=1: cần chẵn. Level=2: cần chia hết 4. Etc.
    required_divisor = 2 ** level
    remainder = n_orig % required_divisor
    if remainder != 0:
        pad_len = required_divisor - remainder
        # 'wrap' mode: lặp tròn → tương đương periodic extension của SWT
        data = np.pad(data, (0, pad_len), mode="wrap")
        logger.debug(
            f"[SWT] Padding {pad_len} sample(s) "
            f"(input len={n_orig}, required divisor={required_divisor})."
        )
    else:
        pad_len = 0

    # ── Áp dụng SWT ──────────────────────────────────────────────────────────
    # pywt.swt trả về list of tuples: [(cAn, cDn), ..., (cA1, cD1)]
    # Thứ tự: highest level trước, lowest level cuối.
    # Với level=1: coeffs = [(cA1, cD1)] → coeffs[-1] = (cA1, cD1)
    # trim_approx=False (default): giữ approximation coefficients
    coeffs = pywt.swt(data, wavelet=wavelet, level=level, trim_approx=False)

    # Lấy level-1 coefficients (phần tử cuối = finest level)
    cA1, cD1 = coeffs[-1]

    # ── Trim về length gốc (nếu đã pad) ──────────────────────────────────────
    if pad_len > 0:
        cA1 = cA1[:n_orig]
        cD1 = cD1[:n_orig]

    assert len(cA1) == n_orig and len(cD1) == n_orig, (
        f"SWT output length mismatch: expected {n_orig}, "
        f"got cA1={len(cA1)}, cD1={len(cD1)}"
    )

    return cA1, cD1


def decompose_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose tất cả input features bằng SWT, giữ Close làm target.

    Features cần decompose: [Open, High, Low, Volume, Deviation].
    Close → giữ nguyên (target, KHÔNG decompose).

    Tham số wavelet lấy từ WAVELET_CONFIG (config.py):
      - wavelet: 'db4'
      - level:   1
      - mode:    ghi chú tham khảo; pywt.swt dùng periodic extension bên trong

    Args:
        df: DataFrame với index=Date và columns bao gồm:
            [Open, High, Low, Close, Volume, Deviation].

    Returns:
        df_wavelet: DataFrame với 11 columns, giữ nguyên DatetimeIndex:
          [Open_Approx,      Open_Detail,
           High_Approx,      High_Detail,
           Low_Approx,       Low_Detail,
           Volume_Approx,    Volume_Detail,
           Deviation_Approx, Deviation_Detail,
           Close]

    Raises:
        KeyError: Nếu thiếu bất kỳ column nào cần thiết.
    """
    # ── Validate input columns ────────────────────────────────────────────────
    required_cols = FEATURES_TO_DECOMPOSE + ["Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"decompose_all_features: thiếu columns {missing}. "
            f"Columns hiện có: {list(df.columns)}"
        )

    # ── Lấy wavelet config từ config.py ──────────────────────────────────────
    wavelet = WAVELET_CONFIG["wavelet"]   # "db4"
    level   = WAVELET_CONFIG["level"]     # 1
    # WAVELET_CONFIG["mode"] = "periodization" → chỉ là tài liệu tham khảo;
    # pywt.swt KHÔNG nhận mode parameter (dùng periodic extension bên trong).

    logger.info(
        f"[SWT] Bắt đầu decompose | wavelet={wavelet} | level={level} | "
        f"features={FEATURES_TO_DECOMPOSE} | rows={len(df):,}"
    )

    # ── Decompose từng feature ────────────────────────────────────────────────
    result: dict[str, np.ndarray] = {}
    for feat in FEATURES_TO_DECOMPOSE:
        cA, cD = apply_swt_to_feature(
            df[feat].values, wavelet=wavelet, level=level
        )
        result[f"{feat}_Approx"] = cA
        result[f"{feat}_Detail"] = cD
        logger.info(
            f"[SWT] {feat:<12} → Approx std={cA.std():.4f}  "
            f"Detail std={cD.std():.6f}"
        )

    # ── Close giữ nguyên làm target ──────────────────────────────────────────
    result["Close"] = df["Close"].values

    # ── Sắp xếp columns đúng thứ tự theo bài báo ─────────────────────────────
    col_order = []
    for feat in FEATURES_TO_DECOMPOSE:
        col_order.append(f"{feat}_Approx")
        col_order.append(f"{feat}_Detail")
    col_order.append("Close")

    df_wavelet = pd.DataFrame(result, index=df.index)[col_order]

    logger.info(
        f"[SWT] Decomposition hoàn tất | "
        f"Input features: {len(FEATURES_TO_DECOMPOSE)} | "
        f"Output: {len(col_order) - 1} features + Close target | "
        f"Shape: {df_wavelet.shape}"
    )
    return df_wavelet


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_wavelet_coefficients(
    df_wavelet: pd.DataFrame,
    ticker: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Vẽ Approximation và Detail Coefficients theo thời gian.
    Tái hiện Fig. 7 (VIC) / Fig. 8 (VCB) trong bài báo Li et al. (2026).

    Layout: 2 subplots side-by-side
      - Left:  Approximation Coefficients (A₁) — 5 features normalized
      - Right: Detail Coefficients (D₁)         — 5 features normalized

    Mỗi feature được normalize min-max về [0, 1] trước khi vẽ để
    so sánh trực quan (các features có đơn vị/scale rất khác nhau).

    Args:
        df_wavelet: DataFrame từ decompose_all_features() với columns
                    [Open_Approx, Open_Detail, ..., Close].
        ticker:     Tên cổ phiếu (dùng trong title và tên file).
        save_path:  Đường dẫn lưu PNG. Mặc định:
                    data/processed/{ticker}_wavelet_plot.png

    Raises:
        KeyError: Nếu thiếu columns cần thiết trong df_wavelet.
    """
    features = ["Open", "High", "Low", "Volume", "Deviation"]

    # Validate columns
    needed = [f"{f}_{s}" for f in features for s in ("Approx", "Detail")]
    missing = [c for c in needed if c not in df_wavelet.columns]
    if missing:
        raise KeyError(f"plot_wavelet_coefficients: thiếu columns {missing}")

    # ── Setup figure ──────────────────────────────────────────────────────────
    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(16, 5))

    # Palette: 5 màu phân biệt tốt
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        """Normalize về [0, 1]; xử lý constant series."""
        lo, hi = arr.min(), arr.max()
        span = hi - lo
        return (arr - lo) / span if span > 1e-10 else np.zeros_like(arr)

    for i, feat in enumerate(features):
        a_norm = _minmax_norm(df_wavelet[f"{feat}_Approx"].values)
        d_norm = _minmax_norm(df_wavelet[f"{feat}_Detail"].values)

        ax_a.plot(
            df_wavelet.index, a_norm,
            color=palette[i], linewidth=0.85, alpha=0.85, label=feat
        )
        ax_d.plot(
            df_wavelet.index, d_norm,
            color=palette[i], linewidth=0.7, alpha=0.75, label=feat
        )

    # ── Titles & labels ───────────────────────────────────────────────────────
    ax_a.set_title(
        f"{ticker} — Approximation Coefficients  A₁\n"
        "Low-frequency trend (db4, level-1)",
        fontsize=11, fontweight="bold"
    )
    ax_d.set_title(
        f"{ticker} — Detail Coefficients  D₁\n"
        "High-frequency volatility (db4, level-1)",
        fontsize=11, fontweight="bold"
    )

    for ax in (ax_a, ax_d):
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Normalized Value [0, 1]", fontsize=9)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.suptitle(
        f"SWT db4 Level-1 Decomposition — {ticker}  (2012–2024)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_path = PROCESSED_DIR / f"{ticker}_wavelet_plot.png"

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info(f"[plot] Wavelet coefficients → {save_path}")
    plt.close(fig)


def visualize_wavelet_functions(
    save_path: str | Path | None = None,
) -> None:
    """
    Vẽ db4 wavelet function (ψ) và scaling function (φ).
    Tái hiện Fig. 6 trong bài báo Li et al. (2026).

    Dùng pywt.Wavelet('db4').wavefun(level=10) để tính:
      - φ (phi): scaling function — hình dạng low-pass filter
      - ψ (psi): wavelet function — mother wavelet (high-pass)
      - x:       grid points (trục X chung, support = [0, 7] cho db4)

    db4 properties:
      - Orthogonal, compact support [0, 7]
      - 4 vanishing moments → capture smooth trends up to cubic

    Args:
        save_path: Đường dẫn lưu PNG. Mặc định:
                   data/processed/db4_wavelet_functions.png
    """
    # ── Tính wavelet & scaling functions ─────────────────────────────────────
    # wavefun(level=10): level quyết định độ mịn của lưới xấp xỉ
    # Returns (phi, psi, x) cho real orthogonal wavelets như db4
    wav = pywt.Wavelet("db4")
    phi, psi, x = wav.wavefun(level=10)

    # ── Setup figure ──────────────────────────────────────────────────────────
    fig, (ax_psi, ax_phi) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: Wavelet function ψ(t) — oscillating, high-pass ──────────────────
    ax_psi.plot(x, psi, color="#1f77b4", linewidth=1.8)
    ax_psi.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.55)
    ax_psi.fill_between(x, psi, 0, alpha=0.10, color="#1f77b4")
    ax_psi.set_title("Wavelet Function  ψ(t)", fontsize=12, fontweight="bold")
    ax_psi.set_xlabel("t", fontsize=11)
    ax_psi.set_ylabel("ψ(t)", fontsize=11)
    ax_psi.set_xlim(x[0], x[-1])
    ax_psi.grid(True, alpha=0.25, linewidth=0.5)
    ax_psi.annotate(
        "db4 — 4 vanishing moments\ncompact support [0, 7]",
        xy=(0.03, 0.97), xycoords="axes fraction",
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.75, edgecolor="silver")
    )

    # ── Right: Scaling function φ(t) — smooth, low-pass ───────────────────────
    ax_phi.plot(x, phi, color="#2ca02c", linewidth=1.8)
    ax_phi.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.55)
    ax_phi.fill_between(x, phi, 0, alpha=0.10, color="#2ca02c")
    ax_phi.set_title("Scaling Function  φ(t)", fontsize=12, fontweight="bold")
    ax_phi.set_xlabel("t", fontsize=11)
    ax_phi.set_ylabel("φ(t)", fontsize=11)
    ax_phi.set_xlim(x[0], x[-1])
    ax_phi.grid(True, alpha=0.25, linewidth=0.5)
    ax_phi.annotate(
        "Daubechies-4 (db4)\northogonal, smooth",
        xy=(0.03, 0.97), xycoords="axes fraction",
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.75, edgecolor="silver")
    )

    plt.suptitle(
        "Daubechies-4 (db4) Wavelet — Scaling & Wavelet Functions",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_path = PROCESSED_DIR / "db4_wavelet_functions.png"

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info(f"[plot] db4 wavelet functions → {save_path}")
    plt.close(fig)