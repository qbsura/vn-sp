/**
 * frontend/js/charts.js
 * =====================
 * Chart rendering functions for VNSP Dashboard — Task 8.2
 *
 * All functions: render*(id, data) — id is a canvas or container element ID.
 * Charts (Chart.js): canvasId → <canvas>
 * HTML renders: containerId → <div> (confusion matrix, correlation, histograms)
 *
 * Dependencies: Chart.js 4.4.2 via CDN (index.html)
 *
 * Public API:
 *   renderPredictedVsActual(canvasId, data)
 *   renderLossCurves(canvasId, data)
 *   renderMSEComparison(canvasId, data)
 *   renderConfusionMatrix(containerId, data)   ← HTML, not canvas
 *   renderROCCurve(canvasId, data)
 *   renderCumulativeReturn(canvasId, data)
 *   renderCorrelationMatrix(containerId, data) ← HTML, not canvas
 *   renderFeatureDistributions(containerId, data)
 */

'use strict';

// =============================================================================
// THEME & PALETTE
// =============================================================================

/** CSS variable → resolved value (dark theme from style.css) */
const C = {
  bg:        '#1e2a45',
  border:    '#2a3a5c',
  text:      '#ffffff',
  muted:     '#a0a0b0',
  accent:    '#e94560',
  blue:      '#64b5f6',
  green:     '#4caf50',
  warning:   '#ff9800',
  purple:    '#ce93d8',
  gridLine:  'rgba(255, 255, 255, 0.06)',
};

/** Per-model colors — consistent across all charts */
const MODEL_COLORS = {
  DNN:    '#e94560',  // accent red
  RNN:    '#64b5f6',  // blue
  GRU:    '#4caf50',  // green
  LSTM:   '#ff9800',  // orange
  BiLSTM: '#ce93d8',  // purple
};

/** Fallback color for unknown model names */
const _modelColor = (name) =>
  MODEL_COLORS[name] || '#a0a0b0';

// =============================================================================
// CHART INSTANCE REGISTRY — cleanup before re-render
// =============================================================================

/** Map: canvasId → Chart instance */
const _charts = {};

/**
 * Destroy existing Chart.js instance on a canvas (if any).
 * Must be called before creating a new chart on the same canvas.
 * @param {string} canvasId
 */
function _destroyChart(canvasId) {
  if (_charts[canvasId]) {
    _charts[canvasId].destroy();
    delete _charts[canvasId];
  }
}

// =============================================================================
// ZOOM & PAN — custom wheel-zoom + drag-pan (không cần plugin ngoài)
// =============================================================================

/**
 * Attach wheel-zoom + drag-pan + double-click-reset cho một Chart.js canvas.
 *
 * Hoạt động với category x-axis (string labels như ngày tháng).
 *   - Mouse wheel UP    : zoom in  (thu hẹp vùng hiển thị, căn giữa con trỏ)
 *   - Mouse wheel DOWN  : zoom out (mở rộng vùng hiển thị)
 *   - Click + drag      : pan trái/phải
 *   - Double-click      : reset về full view
 *
 * Gọi sau khi chart đã được tạo và lưu vào _charts[canvasId].
 *
 * @param {string} canvasId
 */
function _attachZoomPan(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  let dragState = null;  // { startX, startMin, startMax } khi đang drag

  // Helper: lấy [minIdx, maxIdx] hiện tại từ chart options
  function _getRange(chart) {
    const all    = chart.data.labels;
    const minLbl = chart.options.scales?.x?.min;
    const maxLbl = chart.options.scales?.x?.max;
    const minIdx = minLbl != null ? Math.max(0, all.indexOf(minLbl)) : 0;
    let   maxIdx = maxLbl != null ? all.indexOf(maxLbl) : all.length - 1;
    if (maxIdx === -1) maxIdx = all.length - 1;
    return { all, minIdx, maxIdx };
  }

  // ── Wheel Zoom ──────────────────────────────────────────────────────────────
  canvas.addEventListener('wheel', (evt) => {
    evt.preventDefault();
    const chart = _charts[canvasId];
    if (!chart) return;

    const { all, minIdx, maxIdx } = _getRange(chart);
    if (!all.length) return;

    const range    = maxIdx - minIdx;
    // Scroll up = zoom in (nhân 0.80), scroll down = zoom out (nhân 1.25)
    const factor   = evt.deltaY < 0 ? 0.80 : 1.25;
    const newRange = Math.max(20, Math.min(all.length - 1, Math.round(range * factor)));

    // Pivot tại vị trí con trỏ X trên canvas
    const rect     = canvas.getBoundingClientRect();
    const relX     = Math.min(1, Math.max(0, (evt.clientX - rect.left) / rect.width));
    const pivotIdx = minIdx + relX * range;
    let   newMin   = Math.round(pivotIdx - relX * newRange);
    newMin         = Math.max(0, Math.min(all.length - 1 - newRange, newMin));
    const newMax   = Math.min(all.length - 1, newMin + newRange);

    chart.options.scales.x.min = all[newMin];
    chart.options.scales.x.max = all[newMax];
    chart.update('none');
  }, { passive: false });

  // ── Drag Pan ────────────────────────────────────────────────────────────────
  canvas.addEventListener('mousedown', (evt) => {
    const chart = _charts[canvasId];
    if (!chart) return;
    const { all, minIdx, maxIdx } = _getRange(chart);
    dragState = { startX: evt.clientX, startMin: minIdx, startMax: maxIdx };
    canvas.style.cursor = 'grabbing';
  });

  canvas.addEventListener('mousemove', (evt) => {
    if (!dragState) return;
    const chart = _charts[canvasId];
    if (!chart) return;
    const all = chart.data.labels;
    if (!all?.length) return;

    const rect       = canvas.getBoundingClientRect();
    const range      = dragState.startMax - dragState.startMin;
    const pxPerLabel = range > 0 ? rect.width / range : 1;
    const deltaIdx   = Math.round((dragState.startX - evt.clientX) / pxPerLabel);

    let newMin       = Math.max(0, Math.min(all.length - 1 - range, dragState.startMin + deltaIdx));
    const newMax     = Math.min(all.length - 1, newMin + range);

    chart.options.scales.x.min = all[newMin];
    chart.options.scales.x.max = all[newMax];
    chart.update('none');
  });

  const _endDrag = () => { dragState = null; canvas.style.cursor = 'grab'; };
  canvas.addEventListener('mouseup',    _endDrag);
  canvas.addEventListener('mouseleave', _endDrag);

  // ── Double-click: Reset toàn bộ view ─────────────────────────────────────────
  canvas.addEventListener('dblclick', () => {
    const chart = _charts[canvasId];
    if (!chart) return;
    delete chart.options.scales.x.min;
    delete chart.options.scales.x.max;
    chart.update();
    canvas.style.cursor = 'grab';
  });

  canvas.style.cursor = 'grab';
}

// =============================================================================
// SHARED HELPERS
// =============================================================================

/**
 * Show a chart canvas and hide its sibling placeholder.
 * Convention: placeholder ID = canvasId with "chart-" → "" and append "-placeholder"
 *   e.g. "chart-pred-actual" → placeholder div id "pred-chart-placeholder"
 * OR pass explicit placeholderId.
 *
 * @param {string} canvasId
 * @param {string|null} placeholderId
 */
function _showCanvas(canvasId, placeholderId = null) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  canvas.style.display = 'block';

  // Hide placeholder
  const pid = placeholderId ||
    canvasId.replace('chart-', '').replace(/-/g, '-') + '-placeholder';
  const ph = document.getElementById(pid);
  if (ph) ph.classList.add('hidden');
}

/**
 * Base Chart.js options shared across all charts (dark theme).
 * Merge into chart-specific options.
 *
 * @param {object} [extra]  additional options to deep-merge
 * @returns {object}
 */
function _baseOptions(extra = {}) {
  return _merge({
    responsive       : true,
    maintainAspectRatio: true,
    animation        : { duration: 350, easing: 'easeInOutQuart' },
    color            : C.muted,
    font             : { family: "'IBM Plex Mono', monospace", size: 11 },
    plugins: {
      legend: {
        labels: {
          color   : C.muted,
          font    : { family: "'IBM Plex Mono', monospace", size: 11 },
          padding : 16,
          usePointStyle: true,
          pointStyleWidth: 8,
        },
      },
      tooltip: {
        backgroundColor : 'rgba(15, 30, 60, 0.92)',
        titleColor      : C.text,
        bodyColor       : C.muted,
        borderColor     : C.border,
        borderWidth     : 1,
        padding         : 10,
        titleFont       : { family: "'IBM Plex Mono', monospace", size: 11, weight: '600' },
        bodyFont        : { family: "'IBM Plex Mono', monospace", size: 11 },
        cornerRadius    : 5,
        displayColors   : true,
        boxWidth        : 8,
        boxHeight       : 8,
      },
    },
    scales: {
      x: {
        ticks   : { color: C.muted, font: { family: "'IBM Plex Mono', monospace", size: 10 }, maxRotation: 35 },
        grid    : { color: C.gridLine },
        border  : { color: C.border },
      },
      y: {
        ticks   : { color: C.muted, font: { family: "'IBM Plex Mono', monospace", size: 10 } },
        grid    : { color: C.gridLine },
        border  : { color: C.border },
      },
    },
  }, extra);
}

/**
 * Shallow/deep merge helper (replaces lodash.merge for this use case).
 * Recursively merges src into dst without mutating either.
 */
function _merge(dst, src) {
  const out = Object.assign({}, dst);
  for (const k of Object.keys(src)) {
    if (src[k] && typeof src[k] === 'object' && !Array.isArray(src[k])) {
      out[k] = _merge(dst[k] || {}, src[k]);
    } else {
      out[k] = src[k];
    }
  }
  return out;
}

/**
 * Compute histogram bins from an array of numeric values.
 * @param {number[]} values
 * @param {number}   nBins   default 30
 * @returns {{ labels: string[], counts: number[], binWidth: number }}
 */
function _computeHistogram(values, nBins = 30) {
  if (!values || !values.length) return { labels: [], counts: [], binWidth: 0 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return { labels: [String(min)], counts: [values.length], binWidth: 0 };
  const binWidth = (max - min) / nBins;
  const counts   = Array(nBins).fill(0);
  const labels   = Array.from({ length: nBins }, (_, i) =>
    (min + i * binWidth).toPrecision(4)
  );
  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), nBins - 1);
    counts[idx]++;
  }
  return { labels, counts, binWidth };
}

/**
 * Format a number as percentage string.
 * @param {number} part
 * @param {number} total
 * @returns {string}  e.g. "42.3"
 */
function _pct(part, total) {
  return total > 0 ? ((part / total) * 100).toFixed(1) : '0.0';
}

/**
 * Interpolate a color for correlation heatmap.
 * -1 → red, 0 → neutral (#1e2a45), +1 → blue
 * @param {number} v  value in [-1, 1]
 * @returns {string}  CSS rgba color
 */
function _corrToColor(v) {
  const clamped = Math.max(-1, Math.min(1, v));
  if (clamped >= 0) {
    // 0 → neutral, 1 → blue (C.blue = #64b5f6)
    const a = clamped * 0.8;
    return `rgba(100, 181, 246, ${a.toFixed(2)})`;
  } else {
    // -1 → red (C.accent = #e94560), 0 → neutral
    const a = (-clamped) * 0.8;
    return `rgba(233, 69, 96, ${a.toFixed(2)})`;
  }
}

// =============================================================================
// CUSTOM CHART.JS PLUGIN — Vertical best-epoch line
// =============================================================================

/**
 * Inline Chart.js plugin that draws a vertical dashed line at best_epoch.
 * Used by renderLossCurves.
 * Options key: chart.options.plugins.bestEpochLine.epoch (number)
 */
const _bestEpochLinePlugin = {
  id: 'bestEpochLine',
  afterDraw(chart) {
    const opts = chart.options.plugins.bestEpochLine;
    if (!opts || opts.epoch == null) return;

    const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
    const xPos = x.getPixelForValue(opts.epoch);

    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = C.warning;
    ctx.lineWidth   = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.moveTo(xPos, top);
    ctx.lineTo(xPos, bottom);
    ctx.stroke();

    // Label above the line
    ctx.font         = `10px 'IBM Plex Mono', monospace`;
    ctx.fillStyle    = C.warning;
    ctx.globalAlpha  = 1;
    ctx.textAlign    = 'left';
    ctx.fillText(`Best ep:${opts.epoch}`, xPos + 4, top + 13);
    ctx.restore();
  },
};


// =============================================================================
// 1. PREDICTED VS ACTUAL
// =============================================================================

/**
 * Line chart: Actual price (blue) vs Predicted price (red).
 *
 * @param {string} canvasId     ID of <canvas> element
 * @param {object} data
 * @param {string[]}  data.dates     X-axis labels (YYYY-MM-DD or index)
 * @param {number[]}  data.y_true    Actual prices
 * @param {number[]}  data.y_pred    Predicted prices
 * @param {string}   [data.currency] "VND" | "USD"
 * @param {string}   [data.ticker]   e.g. "VCB"
 * @param {string}   [data.placeholderId]
 */
/**
 * Line chart: Predicted vs Actual price.
 *
 * Hỗ trợ 2 formats:
 *   - Multi-model (mới): data = { actual, models:[{name, preds}], dates, ticker, currency }
 *   - Single-model (cũ): data = { y_true, y_pred, dates, ticker, currency }   ← backward-compat
 *
 * @param {string} canvasId
 * @param {object} data
 */
function renderPredictedVsActual(canvasId, data) {
  _destroyChart(canvasId);

  const { currency = 'VND', ticker = '', placeholderId, dates } = data;
  const ctx = document.getElementById(canvasId);
  if (!ctx) { console.warn(`renderPredictedVsActual: canvas #${canvasId} not found`); return; }

  // ── Detect format ────────────────────────────────────────────────────────────
  const isMultiModel = Array.isArray(data.models);
  const actual       = isMultiModel ? data.actual : data.y_true;
  const modelsList   = isMultiModel
    ? data.models
    : [{ name: 'Predicted', preds: data.y_pred }];

  const labels = dates && dates.length
    ? dates
    : Array.from({ length: actual.length }, (_, i) => i + 1);

  const yLabel = currency === 'USD' ? 'Price (USD)' : 'Price (VND)';

  const priceFormat = (v) =>
    currency === 'VND'
      ? Number(v).toLocaleString('vi-VN')
      : Number(v).toFixed(4);

  const opts = _baseOptions({
    plugins: {
      legend: { position: 'top' },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ${ctx.dataset.label}: ${priceFormat(ctx.raw)} ${currency}`,
        },
      },
    },
    scales: {
      x: { ticks: { maxTicksLimit: 12 } },
      y: {
        title: { display: true, text: yLabel, color: C.muted,
          font: { family: "'IBM Plex Mono', monospace", size: 10 } },
      },
    },
  });

  // ── Build datasets ──────────────────────────────────────────────────────────
  // Actual line (trắng, đường đầy)
  const datasets = [
    {
      label          : `Actual (${ticker})`,
      data           : actual,
      borderColor    : '#ffffff',
      backgroundColor: 'transparent',
      borderWidth    : 2,
      pointRadius    : 0,
      tension        : 0.1,
      order          : 99,   // render sau cùng (trên cùng)
    },
  ];

  // Prediction lines — mỗi model 1 màu từ MODEL_COLORS, đường đứt
  modelsList.forEach((m, i) => {
    datasets.push({
      label          : isMultiModel ? m.name : 'Predicted',
      data           : m.preds,
      borderColor    : _modelColor(m.name),
      backgroundColor: 'transparent',
      borderWidth    : 1.5,
      borderDash     : [4, 3],
      pointRadius    : 0,
      tension        : 0.1,
      order          : i,
    });
  });

  _charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: opts,
  });

  _showCanvas(canvasId, placeholderId || 'pred-chart-placeholder');

  // Attach zoom & pan sau khi chart đã render
  _attachZoomPan(canvasId);
}


// =============================================================================
// 2. LOSS CURVES
// =============================================================================

/**
 * Line chart: Train loss (blue) vs Val loss (red) per epoch.
 * Draws a vertical dashed line at best_epoch.
 *
 * @param {string} canvasId
 * @param {object} data
 * @param {number[]}  data.train_losses
 * @param {number[]}  data.val_losses
 * @param {number}   [data.best_epoch]     1-based epoch index
 * @param {boolean}  [data.stopped_early]
 * @param {string}   [data.placeholderId]
 */
function renderLossCurves(canvasId, data) {
  _destroyChart(canvasId);

  const { train_losses, val_losses, best_epoch, stopped_early = false, placeholderId } = data;
  const ctx = document.getElementById(canvasId);
  if (!ctx) { console.warn(`renderLossCurves: canvas #${canvasId} not found`); return; }

  const epochs = Array.from({ length: train_losses.length }, (_, i) => i + 1);

  const opts = _baseOptions({
    plugins: {
      legend    : { position: 'top' },
      bestEpochLine: { epoch: best_epoch },
      tooltip: {
        callbacks: {
          title : (items) => `Epoch ${items[0].label}`,
          label : (ctx)   => ` ${ctx.dataset.label}: ${Number(ctx.raw).toFixed(6)}`,
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Epoch', color: C.muted,
          font: { family: "'IBM Plex Mono', monospace", size: 10 } },
        ticks: { maxTicksLimit: 15 },
      },
      y: {
        title: { display: true, text: 'Loss', color: C.muted,
          font: { family: "'IBM Plex Mono', monospace", size: 10 } },
        type: 'linear',
      },
    },
  });

  const earlyTag = stopped_early ? ' (early stop)' : '';

  _charts[canvasId] = new Chart(ctx, {
    type   : 'line',
    data   : {
      labels  : epochs,
      datasets: [
        {
          label          : `Train Loss${earlyTag}`,
          data           : train_losses,
          borderColor    : C.blue,
          backgroundColor: 'transparent',
          borderWidth    : 1.5,
          pointRadius    : 0,
          tension        : 0.15,
        },
        {
          label          : 'Val Loss',
          data           : val_losses,
          borderColor    : C.accent,
          backgroundColor: 'transparent',
          borderWidth    : 1.5,
          pointRadius    : 0,
          tension        : 0.15,
        },
      ],
    },
    options: opts,
    plugins: [_bestEpochLinePlugin],  // register custom plugin per-chart
  });

  _showCanvas(canvasId, placeholderId || 'loss-placeholder');
}


// =============================================================================
// 3. MSE COMPARISON BAR CHART (Fig. 11)
// =============================================================================

/**
 * Grouped bar chart: Before vs After Wavelet MSE for 5 models.
 * Uses log scale when MSE range spans orders of magnitude (typical for VND).
 *
 * @param {string} canvasId
 * @param {object} data
 * @param {string[]}  data.models               ["DNN","RNN","GRU","LSTM","BiLSTM"]
 * @param {object}    data.before_wavelet        { MSE: number[], MAE: number[], MAPE: number[] }
 * @param {object}    data.after_wavelet         { MSE: number[], MAE: number[], MAPE: number[] }
 * @param {string}   [data.metric]               "MSE" | "MAE" | "MAPE" (default: "MSE")
 * @param {string}   [data.placeholderId]
 */
function renderMSEComparison(canvasId, data) {
  _destroyChart(canvasId);

  const {
    models,
    before_wavelet,
    after_wavelet,
    metric = 'MSE',
    placeholderId,
  } = data;

  const ctx = document.getElementById(canvasId);
  if (!ctx) { console.warn(`renderMSEComparison: canvas #${canvasId} not found`); return; }

  const beforeVals = (before_wavelet[metric] || []).map(v => v ?? null);
  const afterVals  = (after_wavelet[metric]  || []).map(v => v ?? null);

  // Use log scale if max/min ratio > 100 (typical for VND MSE values)
  const allVals = [...beforeVals, ...afterVals].filter(v => v != null && v > 0);
  const useLog  = allVals.length > 1 && (Math.max(...allVals) / Math.min(...allVals)) > 100;

  const opts = _baseOptions({
    plugins: {
      legend: { position: 'top' },
      tooltip: {
        callbacks: {
          label: (ctx) =>
            ` ${ctx.dataset.label}: ${Number(ctx.raw).toLocaleString()}`,
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: C.muted,
          font: { family: "'IBM Plex Mono', monospace", size: 11, weight: '600' },
        },
      },
      y: {
        type : useLog ? 'logarithmic' : 'linear',
        title: {
          display: true,
          text   : `${metric}${useLog ? ' (log scale)' : ''}`,
          color  : C.muted,
          font   : { family: "'IBM Plex Mono', monospace", size: 10 },
        },
        ticks: {
          callback: (v) => Number(v).toLocaleString(),
        },
      },
    },
  });

  _charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels  : models,
      datasets: [
        {
          label          : `Before Wavelet — ${metric}`,
          data           : beforeVals,
          backgroundColor: models.map(m => _modelColor(m) + '55'), // 33% alpha
          borderColor    : models.map(m => _modelColor(m)),
          borderWidth    : 1.5,
          borderRadius   : 3,
        },
        {
          label          : `After Wavelet — ${metric}`,
          data           : afterVals,
          backgroundColor: models.map(m => _modelColor(m) + 'cc'), // 80% alpha
          borderColor    : models.map(m => _modelColor(m)),
          borderWidth    : 1.5,
          borderRadius   : 3,
        },
      ],
    },
    options: opts,
  });

  _showCanvas(canvasId, placeholderId);
}


// =============================================================================
// 4. CONFUSION MATRIX (HTML render — not canvas)
// =============================================================================

/**
 * Render a 2×2 confusion matrix as HTML grid inside containerId.
 * Uses .cm-grid CSS classes from style.css.
 *
 * Matrix layout:  [[TN, FP],
 *                  [FN, TP]]
 * (sklearn convention: row = actual, col = predicted)
 *
 * @param {string} containerId   ID of container <div>
 * @param {object} data          from GET /api/results/{exp_id}/confusion-matrix
 * @param {number[][]}  data.matrix   [[TN,FP],[FN,TP]]
 * @param {string[]}    data.labels   ["DOWN (0)", "UP (1)"]
 * @param {object}      data.stats    {TN, FP, FN, TP, total, accuracy}
 */
function renderConfusionMatrix(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) { console.warn(`renderConfusionMatrix: #${containerId} not found`); return; }

  const { matrix, labels = ['DOWN', 'UP'], stats } = data;
  const TN = stats?.TN ?? matrix[0][0];
  const FP = stats?.FP ?? matrix[0][1];
  const FN = stats?.FN ?? matrix[1][0];
  const TP = stats?.TP ?? matrix[1][1];
  const total = stats?.total ?? (TN + FP + FN + TP);

  // Predicted labels (column headers)
  const predDOWN = labels[0] || 'DOWN';
  const predUP   = labels[1] || 'UP';

  el.innerHTML = `
    <div class="cm-grid">
      <!-- Row 1: empty corner + column labels -->
      <div></div>
      <div class="cm-col-label">Pred: ${predDOWN}</div>
      <div class="cm-col-label">Pred: ${predUP}</div>

      <!-- Row 2: Actual DOWN row -->
      <div class="cm-row-label">Act: ${predDOWN}</div>
      <div class="cm-cell tn">
        <div class="cm-cell-val">${TN.toLocaleString()}</div>
        <div class="cm-cell-label">TN · ${_pct(TN, total)}%</div>
      </div>
      <div class="cm-cell fp">
        <div class="cm-cell-val">${FP.toLocaleString()}</div>
        <div class="cm-cell-label">FP · ${_pct(FP, total)}%</div>
      </div>

      <!-- Row 3: Actual UP row -->
      <div class="cm-row-label">Act: ${predUP}</div>
      <div class="cm-cell fn">
        <div class="cm-cell-val">${FN.toLocaleString()}</div>
        <div class="cm-cell-label">FN · ${_pct(FN, total)}%</div>
      </div>
      <div class="cm-cell tp">
        <div class="cm-cell-val">${TP.toLocaleString()}</div>
        <div class="cm-cell-label">TP · ${_pct(TP, total)}%</div>
      </div>
    </div>
    <div style="text-align:center; margin-top:10px; font-size:11px; color:var(--muted); font-family:var(--font-mono)">
      Total: ${total.toLocaleString()} · Accuracy: ${stats?.accuracy != null ? (stats.accuracy * 100).toFixed(2) : _pct(TN+TP, total)}%
    </div>
  `.trim();
}


// =============================================================================
// 5. ROC CURVE
// =============================================================================

/**
 * Line chart: FPR vs TPR (ROC curve) with dashed diagonal (random classifier).
 *
 * @param {string} canvasId
 * @param {object} data          from GET /api/results/{exp_id}/roc-curve
 * @param {number[]}  data.fpr
 * @param {number[]}  data.tpr
 * @param {number}    data.auc_roc
 * @param {string}   [data.label]        model/exp label for legend
 * @param {string}   [data.placeholderId]
 */
function renderROCCurve(canvasId, data) {
  _destroyChart(canvasId);

  const { fpr, tpr, auc_roc, label = 'Model', placeholderId } = data;
  const ctx = document.getElementById(canvasId);
  if (!ctx) { console.warn(`renderROCCurve: canvas #${canvasId} not found`); return; }

  // Convert parallel arrays to {x,y} pairs for scatter-style line chart
  const rocPoints  = fpr.map((x, i) => ({ x, y: tpr[i] }));
  const diagPoints = [{ x: 0, y: 0 }, { x: 1, y: 1 }];

  const opts = _baseOptions({
    plugins: {
      legend: { position: 'top' },
      tooltip: {
        callbacks: {
          title: () => 'ROC',
          label: (ctx) =>
            ` FPR: ${ctx.raw.x.toFixed(3)}  TPR: ${ctx.raw.y.toFixed(3)}`,
        },
      },
    },
    scales: {
      x: {
        type : 'linear',
        min  : 0,
        max  : 1,
        title: { display: true, text: 'False Positive Rate', color: C.muted,
          font: { family: "'IBM Plex Mono', monospace", size: 10 } },
        ticks: { stepSize: 0.2 },
      },
      y: {
        type : 'linear',
        min  : 0,
        max  : 1,
        title: { display: true, text: 'True Positive Rate', color: C.muted,
          font: { family: "'IBM Plex Mono', monospace", size: 10 } },
        ticks: { stepSize: 0.2 },
      },
    },
    aspectRatio: 1.2,
  });

  _charts[canvasId] = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label          : `${label} (AUC = ${auc_roc.toFixed(4)})`,
          data           : rocPoints,
          borderColor    : C.accent,
          backgroundColor: 'transparent',
          borderWidth    : 2,
          pointRadius    : 0,
          showLine       : true,    // connect scatter points
          tension        : 0,
          fill           : false,
          order          : 0,
        },
        {
          label          : 'Random (AUC = 0.50)',
          data           : diagPoints,
          borderColor    : 'rgba(160, 160, 176, 0.35)',
          backgroundColor: 'transparent',
          borderWidth    : 1.2,
          borderDash     : [7, 5],
          pointRadius    : 0,
          showLine       : true,
          tension        : 0,
          fill           : false,
          order          : 1,
        },
      ],
    },
    options: opts,
  });

  _showCanvas(canvasId, placeholderId || 'roc-placeholder');
}


// =============================================================================
// 6. CUMULATIVE RETURN CHART
// =============================================================================

/**
 * Multi-line or bar chart of trading simulation results.
 *
 * If data.series (time-series daily returns) is provided → line chart.
 * Otherwise, renders a grouped bar chart of cumulative return vs buy-hold
 * for each model (summary mode, using /api/results/trading/{ticker}/{currency}).
 *
 * Time-series mode data shape:
 *   { dates: string[], series: [{label, data: number[], color?}] }
 *
 * Summary bar mode data shape:
 *   { rows: [{model_name, use_wavelet, Cumulative_Return, BuyHold_Return}] }
 *
 * @param {string} canvasId
 * @param {object} data
 * @param {string}  [data.placeholderId]
 */
function renderCumulativeReturn(canvasId, data) {
  _destroyChart(canvasId);

  const ctx = document.getElementById(canvasId);
  if (!ctx) { console.warn(`renderCumulativeReturn: canvas #${canvasId} not found`); return; }

  // ── Time-series mode (dates + series) ────────────────────────────────────
  if (data.series && data.dates) {
    const opts = _baseOptions({
      plugins: { legend: { position: 'top' } },
      scales: {
        x: { ticks: { maxTicksLimit: 10 } },
        y: {
          title: { display: true, text: 'Cumulative Return (%)', color: C.muted,
            font: { family: "'IBM Plex Mono', monospace", size: 10 } },
          ticks: { callback: (v) => `${v.toFixed(1)}%` },
        },
      },
    });

    _charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels  : data.dates,
        datasets: data.series.map((s, i) => ({
          label          : s.label,
          data           : s.data,
          borderColor    : s.color || Object.values(MODEL_COLORS)[i % 5],
          backgroundColor: 'transparent',
          borderWidth    : s.label.includes('Hold') ? 1.5 : 2,
          borderDash     : s.label.includes('Hold') ? [6, 4] : [],
          pointRadius    : 0,
          tension        : 0.1,
        })),
      },
      options: opts,
    });

  // ── Summary bar mode (rows of aggregate metrics) ──────────────────────────
  } else if (data.rows) {
    const rows = data.rows;

    // One "Buy & Hold" value (same for all models — use first row)
    const buyHold = rows.length > 0 ? rows[0].BuyHold_Return ?? rows[0].buyhold_return : 0;

    const barLabels = rows.map(r => {
      const wave = r.use_wavelet ? 'W' : 'N';
      return `${r.model ?? r.model_name ?? '?'} (${wave})`;  // 'model' là field đúng
    });
    const cumReturns = rows.map(r => ((r.Cumulative_Return ?? r.cumulative_return ?? 0) * 100).toFixed(2));
    const buyHoldArr = rows.map(() => ((buyHold) * 100).toFixed(2));
    const modelNames = rows.map(r => r.model ?? r.model_name ?? '');

    const opts = _baseOptions({
      plugins: { legend: { position: 'top' } },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: {
          title: { display: true, text: 'Return (%)', color: C.muted,
            font: { family: "'IBM Plex Mono', monospace", size: 10 } },
          ticks: { callback: (v) => `${v}%` },
        },
      },
    });

    _charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels  : barLabels,
        datasets: [
          {
            label          : 'Strategy Return (%)',
            data           : cumReturns,
            backgroundColor: modelNames.map(m => _modelColor(m) + 'cc'),
            borderColor    : modelNames.map(m => _modelColor(m)),
            borderWidth    : 1.5,
            borderRadius   : 3,
          },
          {
            label          : 'Buy & Hold (%)',
            data           : buyHoldArr,
            backgroundColor: 'rgba(160,160,176,0.25)',
            borderColor    : C.muted,
            borderWidth    : 1.5,
            borderDash     : [4, 4],
            borderRadius   : 3,
            type           : 'line',    // overlay line on bar chart
            pointRadius    : 0,
          },
        ],
      },
      options: opts,
    });

  } else {
    console.warn('renderCumulativeReturn: data must have .series+.dates or .rows');
    return;
  }

  _showCanvas(canvasId, data.placeholderId || 'trading-placeholder');
}


// =============================================================================
// 7. CORRELATION MATRIX (HTML heatmap — not canvas)
// =============================================================================

/**
 * Render a correlation heatmap as an HTML table with color-coded cells.
 * Blue = positive correlation, Red = negative, intensity = |value|.
 *
 * @param {string} containerId   ID of container <div>
 * @param {object} data
 * @param {string[]}   data.labels   feature names
 * @param {number[][]} data.matrix   NxN correlation matrix
 */
function renderCorrelationMatrix(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) { console.warn(`renderCorrelationMatrix: #${containerId} not found`); return; }

  const { labels, matrix } = data;
  const N = labels.length;

  // Build HTML table
  let html = `
    <div style="overflow:auto; max-height:480px;">
    <table style="border-collapse:collapse; font-family:var(--font-mono); font-size:9.5px; width:100%;">
      <thead>
        <tr>
          <th style="padding:4px 6px; color:var(--muted); text-align:left; position:sticky; top:0; background:var(--card-bg); z-index:1;"></th>`;

  for (const lbl of labels) {
    const short = lbl.length > 10 ? lbl.slice(0, 9) + '…' : lbl;
    html += `<th style="padding:4px 6px; color:var(--muted); white-space:nowrap;
                         position:sticky; top:0; background:var(--card-bg); z-index:1;"
                title="${lbl}">${short}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (let i = 0; i < N; i++) {
    html += `<tr>
      <td style="padding:4px 6px; color:var(--muted); white-space:nowrap; font-weight:600;"
          title="${labels[i]}">${labels[i].length > 12 ? labels[i].slice(0, 11) + '…' : labels[i]}</td>`;

    for (let j = 0; j < N; j++) {
      const v    = matrix[i][j];
      const bg   = _corrToColor(v);
      const text = Math.abs(v) > 0.5 ? '#fff' : 'var(--muted)';
      html += `<td style="
          padding: 5px 4px;
          text-align: center;
          background: ${bg};
          color: ${text};
          border: 1px solid rgba(255,255,255,0.03);
          min-width: 36px;"
          title="${labels[i]} vs ${labels[j]}: ${v.toFixed(3)}">
        ${v.toFixed(2)}
      </td>`;
    }
    html += '</tr>';
  }

  html += '</tbody></table></div>';

  // Color scale legend
  html += `
    <div style="display:flex; align-items:center; gap:6px; margin-top:10px;
                font-size:10px; color:var(--muted); font-family:var(--font-mono);">
      <span>−1</span>
      <div style="flex:1; height:8px; border-radius:4px;
                  background:linear-gradient(to right, rgba(233,69,96,0.8), rgba(30,42,69,0), rgba(100,181,246,0.8));"></div>
      <span>+1</span>
    </div>`;

  el.innerHTML = html;
}


// =============================================================================
// 8. FEATURE DISTRIBUTIONS (small multiples histograms)
// =============================================================================

/**
 * Render small multiple histograms for each feature using Chart.js bar charts.
 * Creates a responsive CSS grid of small canvases inside containerId.
 *
 * @param {string} containerId   ID of container <div>
 * @param {object} data
 * @param {Array<{name: string, values: number[]}>} data.features
 * @param {number} [data.nBins=30]  histogram bin count
 */
function renderFeatureDistributions(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) { console.warn(`renderFeatureDistributions: #${containerId} not found`); return; }

  const { features, nBins = 30 } = data;
  if (!features || !features.length) {
    el.innerHTML = '<p style="color:var(--muted);font-size:12px;">No feature data.</p>';
    return;
  }

  // Destroy any previously created histogram charts in this container
  const prevCanvases = el.querySelectorAll('canvas');
  prevCanvases.forEach(c => {
    if (_charts[c.id]) {
      _charts[c.id].destroy();
      delete _charts[c.id];
    }
  });

  // Grid wrapper
  el.innerHTML = `
    <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(200px, 1fr)); gap:14px; padding:4px;">
    ${features.map((f, i) => `
      <div style="background:rgba(255,255,255,0.02); border:1px solid var(--card-border);
                  border-radius:6px; padding:10px 12px;">
        <div style="font-size:10px; font-weight:600; color:var(--muted); margin-bottom:8px;
                    letter-spacing:0.06em; font-family:var(--font-mono);">
          ${f.name}
        </div>
        <canvas id="hist-canvas-${i}" height="80"></canvas>
      </div>`
    ).join('')}
    </div>`;

  // Render individual bar chart for each feature
  for (let i = 0; i < features.length; i++) {
    const { name, values } = features[i];
    const cid = `hist-canvas-${i}`;
    const canvas = document.getElementById(cid);
    if (!canvas) continue;

    const { labels, counts } = _computeHistogram(values, nBins);
    const color = Object.values(MODEL_COLORS)[i % 5];

    _charts[cid] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label          : name,
          data           : counts,
          backgroundColor: color + '88',
          borderColor    : color,
          borderWidth    : 0.8,
          borderRadius   : 1,
        }],
      },
      options: {
        responsive         : true,
        maintainAspectRatio: false,
        animation          : { duration: 200 },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: ([item]) => `~${item.label}`,
              label: ([item]) => ` Count: ${item.raw}`,
            },
          },
        },
        scales: {
          x: {
            display : false,
            grid    : { display: false },
          },
          y: {
            display : false,
            grid    : { display: false },
          },
        },
      },
    });
  }
}


// =============================================================================
// EXPORTS — attach to window for access from main.js and api.js
// =============================================================================

window.VNSP = window.VNSP || {};
Object.assign(window.VNSP, {
  charts: {
    renderPredictedVsActual,
    renderLossCurves,
    renderMSEComparison,
    renderConfusionMatrix,
    renderROCCurve,
    renderCumulativeReturn,
    renderCorrelationMatrix,
    renderFeatureDistributions,
    // Expose internals for testing/debugging
    _destroyChart,
    _charts,
  },
});