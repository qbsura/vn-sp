"""
app/services/preprocessing.py
===============================
Tiền xử lý dữ liệu cổ phiếu cho dự án VNSP.

Module này chứa toàn bộ logic preprocessing theo pipeline bài báo Li et al. (2026):
  Task 2.1 — Feature Derivation (Deviation)            ✅ done
  Task 2.3 — Feature Scaling (FeatureScaler)           ✅ done
  Task 2.4 — Feature Selection by Correlation          ✅ done (file này)
  Task 2.5 — Sequence Builder & Dataset                ← sẽ thêm ở Task 2.5

  Wavelet Decomposition nằm ở app/services/wavelet_service.py (Task 2.2)

Tham chiếu:
  Li et al., "The importance of data noise reduction–wavelet transformation
  in stock price forecasting with Bidirectional LSTM network",
  Engineering Applications of Artificial Intelligence, 165 (2026) 113390.
  Section 3.2.1: Feature Derivation
  Section 3.2.2: Data Scaling (Standard Scaler, Robust Scaler)
  Section 3.3.4: Feature Selection by Correlation (threshold=0.95)
  Appendix A:    PCA (optional, 95% variance)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from app.config import PATHS, SCALER_CONFIG, TARGET_COL

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(PATHS["processed"])


# =============================================================================
# TASK 2.1 — FEATURE DERIVATION
# =============================================================================

def add_deviation_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm feature Deviation = Close - Open vào DataFrame.

    Theo bài báo (Section 3.2.1):
      Deviation đo áp lực mua/bán trong ngày:
        • Deviation > 0: giá đóng cao hơn mở → áp lực mua (bullish)
        • Deviation < 0: giá đóng thấp hơn mở → áp lực bán (bearish)
        • Deviation ≈ 0: cân bằng cung cầu

      Ngưỡng ý nghĩa (thực nghiệm):
        • VND: |Deviation| < 50 VND → biến động nhỏ, ít ý nghĩa dự báo
        • VND: |Deviation| > 50 VND → biến động rõ rệt, phản ánh sentiment
        • USD: |Deviation| < 0.002 USD → tương đương ngưỡng VND tại tỷ giá ~25,000

    Args:
        df: DataFrame với index=Date và các cột OHLCV chuẩn.
            Bắt buộc: có cột 'Close' và 'Open'.

    Returns:
        DataFrame gốc với thêm cột 'Deviation' (float).
        Không thay đổi cột nào khác.

    Raises:
        KeyError: Nếu thiếu cột 'Close' hoặc 'Open'.

    Example:
        >>> df = load_stock_data("VCB")
        >>> df = add_deviation_feature(df)
        >>> df["Deviation"].describe()
    """
    required_cols = ["Close", "Open"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"add_deviation_feature: thiếu cột {missing}. "
            f"Columns hiện có: {list(df.columns)}"
        )

    df = df.copy()  # không mutate DataFrame gốc

    # ── Công thức chính (Eq. 1 trong bài báo) ────────────────────────────────
    # Deviation = Close Price − Open Price
    df["Deviation"] = df["Close"] - df["Open"]

    dev = df["Deviation"]
    logger.info(
        f"[Deviation] Đã thêm feature Deviation | "
        f"Mean: {dev.mean():.2f} | Std: {dev.std():.2f} | "
        f"Min: {dev.min():.2f} | Max: {dev.max():.2f}"
    )

    n_large = (dev.abs() > 50).sum()
    n_total = len(dev)
    logger.info(
        f"[Deviation] Biến động lớn (|Deviation| > 50): "
        f"{n_large:,}/{n_total:,} ngày ({100*n_large/n_total:.1f}%)"
    )

    return df


# =============================================================================
# TASK 2.3 — FEATURE SCALING
# =============================================================================

# ── Helper: resolve scaler type cho một feature ───────────────────────────────
def _resolve_scaler_type(feature: str) -> str:
    """
    Tra SCALER_CONFIG để xác định loại scaler cho feature.

    Returns:
        "standard" | "robust" | "none"
        Mặc định "standard" cho feature không có trong bất kỳ nhóm nào.
    """
    if feature in SCALER_CONFIG["none"]:
        return "none"
    if feature in SCALER_CONFIG["robust"]:
        return "robust"
    # Explicit standard list → hoặc fallback default → dùng standard
    return "standard"


class FeatureScaler:
    """
    Scaler tổng hợp cho toàn bộ features của VNSP pipeline.

    Phân công scaler theo bài báo (Section 3.2.2) và SCALER_CONFIG (config.py):
      - StandardScaler : features phân phối gần chuẩn
                         (Detail coefficients, Deviation_Approx/Detail, giá raw)
      - RobustScaler   : features có outlier/lệch (Volume, High/Low Approx)
      - Không scale    : Open_Approx (giữ trend information), Open (no-wavelet)

    Nguyên tắc chống data leakage:
      - fit() và fit_transform() CHỈ được gọi trên training data.
      - transform() áp dụng scalers đã fit lên val/test data.
      - Mỗi Walk-Forward fold nên tạo một instance FeatureScaler riêng.

    Target column (Close):
      - Luôn được fit với StandardScaler để hỗ trợ inverse_transform_target().
      - Cần thiết cho regression: predictions (scaled) → giá thực (VND/USD).
      - Trong no-wavelet case: "Close" đã có trong SCALER_CONFIG["standard"],
        nên không bị fit trùng.

    Usage:
        scaler = FeatureScaler()
        df_train_scaled = scaler.fit_transform(df_train)
        df_test_scaled  = scaler.transform(df_test)
        price_real = scaler.inverse_transform_target(model_predictions)
    """

    def __init__(self) -> None:
        # {feature_name: fitted sklearn scaler object}
        self.scalers: dict = {}
        self._fitted: bool = False

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame) -> "FeatureScaler":
        """
        Fit scalers CHỈ trên training data để tránh data leakage.

        Logic phân công:
          1. Duyệt tất cả columns trong df_train.
          2. Tra SCALER_CONFIG để biết loại scaler phù hợp.
          3. TARGET_COL ("Close") luôn được fit StandardScaler riêng
             (cần cho inverse_transform_target, kể cả wavelet case).

        Args:
            df_train: Training DataFrame (đã qua wavelet hoặc raw OHLCV+Deviation).

        Returns:
            self (để chain: scaler.fit(df).transform(df))
        """
        self.scalers = {}

        for col in df_train.columns:
            scaler_type = _resolve_scaler_type(col)

            if scaler_type == "none":
                # Không scale — chỉ log, không tạo scaler
                logger.debug(f"[Scaler] {col}: KHÔNG scale (giữ trend info)")
                continue

            # Reshape về 2-D vì sklearn yêu cầu (n_samples, 1)
            values = df_train[col].values.reshape(-1, 1)

            if scaler_type == "robust":
                sc = RobustScaler()
            else:
                # "standard" hoặc default fallback
                sc = StandardScaler()

            sc.fit(values)
            self.scalers[col] = sc
            logger.debug(f"[Scaler] {col}: {sc.__class__.__name__} fitted")

        # ── Đảm bảo Close luôn có scaler (wavelet case không có Close trong SCALER_CONFIG) ──
        if TARGET_COL not in self.scalers and TARGET_COL in df_train.columns:
            sc_close = StandardScaler()
            sc_close.fit(df_train[TARGET_COL].values.reshape(-1, 1))
            self.scalers[TARGET_COL] = sc_close
            logger.debug(
                f"[Scaler] {TARGET_COL}: StandardScaler fitted "
                "(wavelet case — needed for inverse_transform_target)"
            )

        self._fitted = True
        logger.info(
            f"[FeatureScaler] fit() xong | "
            f"{len(self.scalers)} scalers fitted | "
            f"Columns không scale: "
            f"{[c for c in df_train.columns if c not in self.scalers]}"
        )
        return self

    # ── transform ─────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng scalers đã fit lên DataFrame (train hoặc test/val).

        Chỉ scale những columns có trong self.scalers.
        Columns không trong scalers (ví dụ Open_Approx, Open) giữ nguyên.

        Args:
            df: DataFrame cần scale. Phải có cùng structure với df_train khi fit.

        Returns:
            DataFrame mới (copy) với các columns đã được scale.

        Raises:
            RuntimeError: Nếu chưa gọi fit() trước.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler.transform(): chưa fit! Gọi fit() hoặc "
                "fit_transform() với training data trước."
            )

        df_out = df.copy()

        for col, sc in self.scalers.items():
            if col not in df_out.columns:
                logger.warning(
                    f"[FeatureScaler.transform] Column '{col}' không có trong df, bỏ qua."
                )
                continue
            df_out[col] = sc.transform(
                df_out[col].values.reshape(-1, 1)
            ).flatten()

        return df_out

    # ── fit_transform ─────────────────────────────────────────────────────────
    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit trên df_train rồi transform ngay df_train.

        Convenience method = fit() + transform() trên cùng một DataFrame.
        CHỈ dùng cho training data.

        Args:
            df_train: Training DataFrame.

        Returns:
            DataFrame đã scale.
        """
        self.fit(df_train)
        return self.transform(df_train)

    # ── inverse_transform_target ──────────────────────────────────────────────
    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions/target về giá thực (VND hoặc USD).

        Chỉ áp dụng cho TARGET_COL ("Close").
        Dùng khi cần so sánh predicted price với giá thực để tính metrics.

        Args:
            values: Array 1-D hoặc 2-D của predicted/actual scaled Close values.

        Returns:
            Array cùng shape, đơn vị giá gốc (VND hoặc USD).

        Raises:
            RuntimeError: Nếu chưa fit hoặc không có scaler cho Close.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler.inverse_transform_target(): chưa fit!"
            )
        if TARGET_COL not in self.scalers:
            raise RuntimeError(
                f"Không có scaler cho '{TARGET_COL}'. "
                "Kiểm tra lại: Close có trong df_train khi fit() không?"
            )

        arr = np.asarray(values, dtype=np.float64)
        orig_shape = arr.shape
        result = self.scalers[TARGET_COL].inverse_transform(
            arr.reshape(-1, 1)
        ).flatten()
        return result.reshape(orig_shape)

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """
        Lưu toàn bộ scaler vào file pickle.

        Args:
            path: Đường dẫn file .pkl (sẽ tạo thư mục cha nếu chưa có).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"[FeatureScaler] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        """
        Load FeatureScaler từ file pickle.

        Args:
            path: Đường dẫn file .pkl đã lưu bằng save().

        Returns:
            FeatureScaler đã fit (ready to transform).

        Raises:
            FileNotFoundError: Nếu file không tồn tại.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy scaler file: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"[FeatureScaler] Loaded ← {path} | scalers: {list(obj.scalers.keys())}")
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"FeatureScaler({status}, "
            f"n_scalers={len(self.scalers)}, "
            f"scaled={list(self.scalers.keys())})"
        )


# =============================================================================
# CONVENIENCE FUNCTION — NO-WAVELET CASE
# =============================================================================

def scale_features_no_wavelet(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], FeatureScaler]:
    """
    Áp dụng scaling cho case KHÔNG dùng wavelet (raw OHLCV + Deviation).

    Phân công scaler cho no-wavelet features (Section 3.2.2 + SCALER_CONFIG):
      - Open:             KHÔNG scale (giữ trend information)
      - High, Low:        StandardScaler
      - Close:            StandardScaler (target; cần cho inverse_transform)
      - Volume:           RobustScaler (skewed distribution, outliers)
      - Deviation:        StandardScaler

    Args:
        df_train: Training DataFrame với columns OHLCV + Deviation.
                  Bắt buộc có: [Open, High, Low, Close, Volume, Deviation].
        df_test:  (Optional) Test/val DataFrame. Nếu truyền vào, transform
                  bằng scaler đã fit trên train (chống data leakage).

    Returns:
        (df_train_scaled, df_test_scaled, scaler)
        df_test_scaled là None nếu df_test=None.

    Example:
        >>> df_train_s, df_test_s, sc = scale_features_no_wavelet(df_train, df_test)
        >>> predictions_real = sc.inverse_transform_target(model_preds)
    """
    required = ["Open", "High", "Low", "Close", "Volume", "Deviation"]
    missing = [c for c in required if c not in df_train.columns]
    if missing:
        raise KeyError(f"scale_features_no_wavelet: thiếu columns {missing}")

    scaler = FeatureScaler()
    df_train_scaled = scaler.fit_transform(df_train)

    df_test_scaled = scaler.transform(df_test) if df_test is not None else None

    logger.info(
        f"[scale_no_wavelet] Train: {df_train.shape} → scaled. "
        + (f"Test: {df_test.shape} → scaled." if df_test is not None else "No test set.")
    )
    return df_train_scaled, df_test_scaled, scaler


# =============================================================================
# VISUALIZATION — FEATURE DISTRIBUTIONS (Fig. 3 & 4)
# =============================================================================

def plot_feature_distributions(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    ticker: str,
    save_path: Optional[str | Path] = None,
    max_features: int = 6,
) -> None:
    """
    Vẽ distribution (histogram + KDE) trước và sau scaling.
    Tái hiện Fig. 3 (before) và Fig. 4 (after) trong bài báo Li et al. (2026).

    Layout:
      - Row 0 (top):    Before scaling — histogram + KDE
      - Row 1 (bottom): After scaling  — histogram + KDE
      - Columns: mỗi column là 1 feature

    Features hiển thị: ưu tiên các features nổi bật (Approx/Detail, Volume)
    để thể hiện sự khác biệt distribution trước/sau. Tối đa max_features columns.

    Args:
        df_before:    DataFrame trước khi scale (raw / wavelet coefficients).
        df_after:     DataFrame sau khi scale (cùng columns, đã scale).
        ticker:       Tên cổ phiếu (dùng trong title).
        save_path:    Đường dẫn lưu PNG. Mặc định:
                      data/processed/{ticker}_distributions.png
        max_features: Số features tối đa hiển thị (default=6).

    Raises:
        ValueError: Nếu df_before và df_after có columns khác nhau.
    """
    # ── Chọn features để plot ─────────────────────────────────────────────────
    # Ưu tiên theo thứ tự: Approx features, Detail features, raw features
    PRIORITY_COLS = [
        "Open_Approx", "Open_Detail",
        "Volume_Approx", "Volume_Detail",
        "Deviation_Approx", "Deviation_Detail",
        "Open", "High", "Volume", "Deviation",
        "Close", "Low",
    ]
    all_cols = [c for c in df_before.columns if c != TARGET_COL]
    # Sắp xếp theo priority, sau đó thêm các cols còn lại
    ordered = [c for c in PRIORITY_COLS if c in all_cols]
    ordered += [c for c in all_cols if c not in ordered]
    plot_cols = ordered[:max_features]

    n_cols = len(plot_cols)
    if n_cols == 0:
        logger.warning("[plot_distributions] Không có feature nào để plot (ngoài Close).")
        return

    # ── Setup figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(3.2 * n_cols, 6),
        constrained_layout=True,
    )
    # Đảm bảo axes luôn là 2-D array (kể cả khi n_cols=1)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["Before Scaling", "After Scaling"]
    row_data   = [df_before, df_after]
    row_color  = ["#4C72B0", "#55A868"]   # xanh đậm / xanh lá

    for row_idx, (label, df_row, color) in enumerate(
        zip(row_labels, row_data, row_color)
    ):
        for col_idx, feat in enumerate(plot_cols):
            ax = axes[row_idx, col_idx]

            if feat not in df_row.columns:
                ax.set_visible(False)
                continue

            vals = df_row[feat].dropna().values

            # ── Histogram ────────────────────────────────────────────────────
            ax.hist(
                vals, bins=40, density=True,
                color=color, alpha=0.4, edgecolor="none"
            )

            # ── KDE overlay ──────────────────────────────────────────────────
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                x_range = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(x_range, kde(x_range), color=color, linewidth=1.8)
            except Exception:
                pass  # KDE có thể fail nếu data constant sau scaling

            # ── Labels ───────────────────────────────────────────────────────
            # Chỉ show xlabel ở row dưới, ylabel ở col đầu
            if row_idx == 1:
                ax.set_xlabel(feat, fontsize=8.5)
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=8.5)

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, linewidth=0.4)

            # Title chỉ ở row trên
            if row_idx == 0:
                ax.set_title(feat, fontsize=9, fontweight="bold", pad=4)

        # Row label bên trái
        axes[row_idx, 0].set_ylabel(
            f"Density\n({label})", fontsize=8.5, labelpad=6
        )

    plt.suptitle(
        f"{ticker} — Feature Distributions Before & After Scaling",
        fontsize=12, fontweight="bold", y=1.01
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_path = PROCESSED_DIR / f"{ticker}_distributions.png"

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info(f"[plot] Feature distributions → {save_path}")
    plt.close(fig)

# =============================================================================
# TASK 2.4 — FEATURE SELECTION BY CORRELATION
# =============================================================================

def select_features_by_correlation(
    df: pd.DataFrame,
    threshold: float = 0.95,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Loại bỏ features có tương quan cao với nhau để giảm redundancy.

    Thuật toán (Section 3.3.4):
      1. Tính Pearson correlation matrix cho tất cả features (không tính target).
      2. Xét upper triangle (từng cặp feature một lần duy nhất).
      3. Với mỗi cặp (i, j) có |corr(i, j)| > threshold:
           → Loại feature nào có |corr với target| THẤP HƠN.
           → Nếu bằng nhau → loại feature j (convention).
      4. Dùng greedy: feature đã bị loại không được xét thêm.

    Kết quả phụ thuộc vào thứ tự duyệt column → stable với input cố định.

    Args:
        df:         DataFrame đã scaling, bao gồm cả target_col.
                    Columns: input features + target_col.
        threshold:  Ngưỡng |Pearson correlation| để loại bỏ (default: 0.95).
                    Lấy từ CORRELATION_THRESHOLD trong config.py.
        target_col: Tên cột target (default: 'Close').

    Returns:
        (df_selected, dropped_features)
        - df_selected:      DataFrame chỉ còn features được giữ + target_col.
        - dropped_features: List tên features đã loại (có thể rỗng nếu không
                            có cặp nào vượt threshold).

    Raises:
        KeyError: Nếu target_col không có trong df.

    Example:
        >>> df_selected, dropped = select_features_by_correlation(df_wavelet)
        >>> print(f"Kept: {df_selected.columns.tolist()}")
    """
    if target_col not in df.columns:
        raise KeyError(
            f"select_features_by_correlation: '{target_col}' không có trong df. "
            f"Columns: {list(df.columns)}"
        )

    # ── Tách features khỏi target ─────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != target_col]

    if len(feature_cols) < 2:
        logger.info("[CorrelationSelect] Chỉ có < 2 features, bỏ qua selection.")
        return df.copy(), []

    # ── Tính correlation matrix giữa các features ─────────────────────────────
    # Chỉ dùng Pearson (linear correlation, phù hợp với giá cổ phiếu)
    feat_corr = df[feature_cols].corr(method="pearson").abs()

    # ── |Correlation| của từng feature với target ─────────────────────────────
    target_corr_abs = df[feature_cols].corrwith(df[target_col]).abs()

    # ── Upper triangle mask (trên đường chéo chính, k=1) ─────────────────────
    # Chỉ xét mỗi cặp (i, j) một lần: i < j theo thứ tự columns
    upper = feat_corr.where(
        np.triu(np.ones(feat_corr.shape, dtype=bool), k=1)
    )

    # ── Greedy selection ──────────────────────────────────────────────────────
    to_drop: set[str] = set()

    for col in feature_cols:
        if col in to_drop:
            continue  # col này đã bị loại ở lần trước

        col_target_corr = target_corr_abs.get(col, 0.0)

        for row in feature_cols:
            if row == col or row in to_drop:
                continue  # bỏ qua diagonal và đã-loại

            pair_corr = upper.loc[row, col]
            if pd.isna(pair_corr) or pair_corr <= threshold:
                continue  # cặp này không vi phạm → giữ cả hai

            # ── Cặp (row, col) vi phạm ngưỡng correlation ────────────────────
            row_target_corr = target_corr_abs.get(row, 0.0)

            if row_target_corr <= col_target_corr:
                # row ít liên quan đến target hơn → loại row
                to_drop.add(row)
                logger.debug(
                    f"[CorrelationSelect] Drop '{row}' "
                    f"(corr_target={row_target_corr:.4f}) — high corr "
                    f"with '{col}' (={pair_corr:.4f}), "
                    f"keep '{col}' (corr_target={col_target_corr:.4f})"
                )
            else:
                # col ít liên quan hơn → loại col, dừng xét rows cho col này
                to_drop.add(col)
                logger.debug(
                    f"[CorrelationSelect] Drop '{col}' "
                    f"(corr_target={col_target_corr:.4f}) — high corr "
                    f"with '{row}' (={pair_corr:.4f}), "
                    f"keep '{row}' (corr_target={row_target_corr:.4f})"
                )
                break  # col đã bị loại, không cần xét thêm rows cho col này

    # ── Kết quả ───────────────────────────────────────────────────────────────
    dropped_features = sorted(to_drop)
    kept_features    = [c for c in feature_cols if c not in to_drop]

    logger.info(
        f"[CorrelationSelect] threshold={threshold} | "
        f"Input: {len(feature_cols)} features | "
        f"Dropped ({len(dropped_features)}): {dropped_features} | "
        f"Kept ({len(kept_features)}): {kept_features}"
    )

    # Giữ đúng thứ tự columns gốc
    final_cols = [c for c in df.columns if c not in to_drop]
    return df[final_cols].copy(), dropped_features


# =============================================================================
# VISUALIZATION — CORRELATION MATRIX (Fig. 10)
# =============================================================================

def plot_correlation_matrix(
    df: pd.DataFrame,
    ticker: str,
    save_path: Optional[str | Path] = None,
    target_col: str = TARGET_COL,
) -> None:
    """
    Vẽ heatmap Pearson correlation matrix của các features.
    Tái hiện Fig. 10 trong bài báo Li et al. (2026).

    Visualize tương quan giữa các features để:
      - Xác nhận kết quả select_features_by_correlation().
      - Nhận diện cặp features nào bị loại.

    Args:
        df:         DataFrame gồm features + target_col.
        ticker:     Tên cổ phiếu (dùng trong title và tên file).
        save_path:  Đường dẫn PNG. Mặc định:
                    data/processed/{ticker}_correlation_matrix.png
        target_col: Cột target — được đặt cuối nếu có trong df.

    Notes:
        figsize tự điều chỉnh: max(8, 0.7 * n_features) per side.
        Colormap: coolwarm, center=0 (đỏ = tương quan dương, xanh = âm).
        Annotation: 1 decimal place.
    """
    # ── Sắp xếp: features trước, target sau ──────────────────────────────────
    feat_cols = [c for c in df.columns if c != target_col]
    plot_cols = feat_cols + ([target_col] if target_col in df.columns else [])
    n = len(plot_cols)

    if n < 2:
        logger.warning("[plot_corr_matrix] Cần ít nhất 2 columns để vẽ heatmap.")
        return

    # ── Tính correlation matrix (dùng giá trị signed, không abs) ─────────────
    corr = df[plot_cols].corr(method="pearson")

    # ── figsize tự điều chỉnh theo số features ───────────────────────────────
    cell_size = max(0.65, 7.5 / n)   # nhỏ hơn nếu nhiều features
    fig_size  = max(8.0, cell_size * n)
    fig, ax   = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    # ── Seaborn heatmap ───────────────────────────────────────────────────────
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".1f",                      # 1 decimal place như bài báo
        annot_kws={"size": max(6, 9 - n // 3)},
        linewidths=0.4,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson correlation"},
    )

    # ── Highlight threshold line (nếu muốn) ───────────────────────────────────
    # Không vẽ thêm để giữ layout gọn như bài báo

    ax.set_title(
        f"{ticker} — Feature Correlation Matrix (Pearson)",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.tick_params(axis="x", rotation=45, labelsize=max(7, 10 - n // 4))
    ax.tick_params(axis="y", rotation=0,  labelsize=max(7, 10 - n // 4))

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if save_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_path = PROCESSED_DIR / f"{ticker}_correlation_matrix.png"

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info(f"[plot] Correlation matrix → {save_path}")
    plt.close(fig)


# =============================================================================
# OPTIONAL — PCA (sau feature selection)
# =============================================================================

def apply_pca(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    variance_threshold: float = 0.95,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], PCA, int]:
    """
    Áp dụng PCA để giảm chiều sau feature selection.

    Ghi chú (Appendix A, project guide):
      PCA là OPTIONAL trong pipeline VNSP. Dùng sau feature selection khi
      vẫn còn nhiều features hoặc khi muốn visualize dữ liệu 2-D/3-D.
      Thứ tự pipeline chuẩn:
        wavelet → scaling → feature selection → [PCA optional] → sequences

    Nguyên tắc chống data leakage:
      PCA được fit CHỈ trên df_train.
      df_test được transform bằng PCA đã fit trên train.

    Args:
        df_train:           Training DataFrame, bao gồm target_col.
        df_test:            (Optional) Test DataFrame, cùng structure df_train.
        variance_threshold: Giữ tối thiểu components để explain tỷ lệ này.
                            sklearn PCA chấp nhận float ∈ (0, 1).
                            Default: 0.95 (giữ 95% variance, Appendix A).
        target_col:         Cột target — không đưa vào PCA, append lại sau.

    Returns:
        (df_train_pca, df_test_pca, pca_obj, n_components)
        - df_train_pca:  DataFrame train sau PCA, columns=[PC1, PC2, ..., Close].
        - df_test_pca:   DataFrame test sau PCA (None nếu df_test=None).
        - pca_obj:       Fitted sklearn PCA object (để transform data mới).
        - n_components:  Số components được giữ.

    Raises:
        KeyError: Nếu target_col không có trong df_train.
        ValueError: Nếu không có features nào (chỉ có target_col).

    Example:
        >>> df_tr_pca, df_te_pca, pca, k = apply_pca(df_train, df_test)
        >>> print(f"PCA giữ {k} components cho {variance_threshold*100:.0f}% variance")
    """
    if target_col not in df_train.columns:
        raise KeyError(
            f"apply_pca: '{target_col}' không có trong df_train. "
            f"Columns: {list(df_train.columns)}"
        )

    # ── Tách features khỏi target ─────────────────────────────────────────────
    feature_cols = [c for c in df_train.columns if c != target_col]
    if len(feature_cols) == 0:
        raise ValueError("apply_pca: không có features nào (chỉ có target_col).")

    X_train = df_train[feature_cols].values  # shape: (n_train, n_features)

    # ── Fit PCA trên train only ───────────────────────────────────────────────
    # n_components = float ∈ (0,1): sklearn tự chọn số components tối thiểu
    # để explain >= variance_threshold của variance
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_train_pca = pca.fit_transform(X_train)

    n_components = pca.n_components_
    explained    = pca.explained_variance_ratio_.sum()

    logger.info(
        f"[PCA] fit on train | "
        f"Input features: {len(feature_cols)} | "
        f"Components kept: {n_components} | "
        f"Variance explained: {explained:.4f} ({explained*100:.1f}%)"
    )

    # ── Tạo DataFrame train PCA ───────────────────────────────────────────────
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    df_train_pca = pd.DataFrame(X_train_pca, index=df_train.index, columns=pc_cols)
    df_train_pca[target_col] = df_train[target_col].values

    # ── Transform test nếu có ─────────────────────────────────────────────────
    df_test_pca: Optional[pd.DataFrame] = None
    if df_test is not None:
        # Validate test có đủ columns
        missing_test = [c for c in feature_cols if c not in df_test.columns]
        if missing_test:
            raise KeyError(
                f"apply_pca: df_test thiếu columns {missing_test}. "
                "Train và test phải có cùng feature columns."
            )
        X_test = df_test[feature_cols].values
        X_test_pca = pca.transform(X_test)   # dùng PCA đã fit trên train
        df_test_pca = pd.DataFrame(X_test_pca, index=df_test.index, columns=pc_cols)
        if target_col in df_test.columns:
            df_test_pca[target_col] = df_test[target_col].values
        logger.info(
            f"[PCA] transform test | "
            f"Test shape: {df_test.shape} → {df_test_pca.shape}"
        )

    return df_train_pca, df_test_pca, pca, n_components