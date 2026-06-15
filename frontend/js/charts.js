/**
 * frontend/js/charts.js
 * ======================
 * Chart building functions for VNSP Dashboard.
 * Redesign 2026-06: Added ROC overlay, Trading line chart, enhanced axis zoom.
 *
 * Exports via window.VNSP.charts:
 *   buildPredChart(canvasId, data)       — Pred vs Actual line chart
 *   buildLossChart(canvasId, data)       — Train/Val loss curves
 *   buildTradingLineChart(canvasId, ts)  — Cumulative return over time (NEW)
 *   buildRocOverlayChart(canvasId, rocs) — 5-model ROC overlay (NEW)
 *   destroyChart(canvasId)               — Destroy existing chart instance
 */

'use strict';

// ── Active chart instances (canvasId → Chart instance) ──────────────────────
const _chartInstances = {};

/** Destroy an existing Chart.js instance on a canvas (prevents redraw errors). */
function destroyChart(canvasId) {
  if (_chartInstances[canvasId]) {
    _chartInstances[canvasId].destroy();
    delete _chartInstances[canvasId];
  }
}

// ── Color palette for 5 models ────────────────────────────────────────────────
const MODEL_COLORS = {
  BiLSTM: '#64b5f6',  // blue
  LSTM  : '#ffa726',  // amber
  GRU   : '#66bb6a',  // green
  RNN   : '#ef5350',  // red
  DNN   : '#ce93d8',  // purple
};
const BUY_HOLD_COLOR = '#90a4ae'; // grey-blue for Buy & Hold baseline

// ── Shared Chart.js defaults ────────────────────────────────────────────────
const BASE_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 300 },
  plugins: {
    legend: { labels: { color: '#e0e0e0', font: { size: 11 }, boxWidth: 14 } },
    tooltip: {
      backgroundColor: '#1c1c35', titleColor: '#64b5f6', bodyColor: '#e0e0e0',
      borderColor: '#3a3a5a', borderWidth: 1,
    },
  },
  scales: {
    x: {
      ticks: { color: '#9e9e9e', font: { size: 10 } },
      grid:  { color: 'rgba(255,255,255,.06)' },
    },
    y: {
      ticks: { color: '#9e9e9e', font: { size: 10 } },
      grid:  { color: 'rgba(255,255,255,.06)' },
    },
  },
};

// =============================================================================
// CUSTOM ZOOM/PAN — không dùng chartjs-plugin-zoom (CDN không ổn định)
// Thiết kế: Google Maps style
//   - Wheel: zoom vào/ra tại VỊ TRÍ CON TRỎ (điểm neo cố định)
//   - Drag (mousedown + mousemove): pan/kéo thả
//   - Dblclick: reset về toàn bộ data
// =============================================================================

/**
 * Lấy current visible range của scale.
 * Chart.js category scale: min/max là index (0..N-1).
 * Chart.js linear scale: min/max là giá trị thực.
 */
function _getRange(scale, fallbackMax) {
  const min = (scale.min !== undefined && scale.min !== null && !isNaN(Number(scale.min)))
    ? Number(scale.min) : 0;
  const max = (scale.max !== undefined && scale.max !== null && !isNaN(Number(scale.max)))
    ? Number(scale.max) : fallbackMax;
  return { min, max };
}

/**
 * Attach custom wheel-zoom + drag-pan lên một Chart.js chart.
 * Không cần plugin ngoài — hoạt động với Chart.js 4.x thuần.
 *
 * @param {Chart}    chart - Chart.js instance
 * @param {'x'|'xy'} mode - 'x' = zoom X axis only, 'xy' = cả 2 trục
 */
function _attachZoomPan(chart, mode = 'x') {
  const canvas = chart.canvas;
  if (!canvas) return;

  // Số điểm data trên X (dùng làm bound khi clamp)
  const xDataMax = () =>
    Math.max(0, (chart.data.labels?.length ?? chart.data.datasets[0]?.data?.length ?? 1) - 1);

  let isDragging = false;
  let dragStart  = { x: 0, y: 0 };
  let dragRange  = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };

  // ── WHEEL ZOOM — tại vị trí con trỏ ────────────────────────────────────────
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();

    const rect      = canvas.getBoundingClientRect();
    const chartArea = chart.chartArea;
    // scroll down = zoom out (factor >1), scroll up = zoom in (factor <1)
    const factor    = e.deltaY > 0 ? 1.15 : 0.87;

    // ── X axis ────────────────────────────────────────────────────────────────
    const xN    = xDataMax();
    const xScl  = chart.scales.x;
    const { min: xMin, max: xMax } = _getRange(xScl, xN);
    const xRange = xMax - xMin;

    // Tỉ lệ cursor trong chartArea [0..1]
    const xRatio  = Math.max(0, Math.min(1,
      (e.clientX - rect.left - chartArea.left) / chartArea.width
    ));
    // Điểm neo: giữ cố định data value tại cursor khi zoom
    const xAnchor   = xMin + xRatio * xRange;
    const newXRange = xRange * factor;

    chart.options.scales.x.min = Math.max(0,  xAnchor - xRatio       * newXRange);
    chart.options.scales.x.max = Math.min(xN, xAnchor + (1 - xRatio) * newXRange);

    // ── Y axis (chỉ khi mode='xy') ────────────────────────────────────────────
    if (mode === 'xy') {
      const yScl  = chart.scales.y;
      const { min: yMin, max: yMax } = _getRange(yScl, 1);
      const yRange = yMax - yMin;

      const yRatio  = Math.max(0, Math.min(1,
        (e.clientY - rect.top - chartArea.top) / chartArea.height
      ));
      // Y pixel tăng xuống dưới, data tăng lên trên → invert ratio
      const yAnchor   = yMax - yRatio * yRange;
      const newYRange = yRange * factor;

      chart.options.scales.y.min = Math.max(0, yAnchor - (1 - yRatio) * newYRange);
      chart.options.scales.y.max = Math.min(1, yAnchor + yRatio       * newYRange);
    }

    chart.update('none');  // 'none' = bỏ animation → mượt
  }, { passive: false });

  // ── DRAG PAN — kéo thả như Google Maps ────────────────────────────────────
  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;  // chỉ left-click
    isDragging     = true;
    dragStart      = { x: e.clientX, y: e.clientY };
    const xN       = xDataMax();
    const { min: xMin, max: xMax } = _getRange(chart.scales.x, xN);
    dragRange.xMin = xMin;
    dragRange.xMax = xMax;
    if (mode === 'xy') {
      const { min: yMin, max: yMax } = _getRange(chart.scales.y, 1);
      dragRange.yMin = yMin;
      dragRange.yMax = yMax;
    }
    canvas.style.cursor = 'grabbing';
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const chartArea = chart.chartArea;
    const xN        = xDataMax();

    // Pan X: delta pixel → delta data units
    const dx      = e.clientX - dragStart.x;
    const xRange  = dragRange.xMax - dragRange.xMin;
    const xDelta  = -(dx / chartArea.width) * xRange;
    let newXMin   = dragRange.xMin + xDelta;
    let newXMax   = dragRange.xMax + xDelta;

    // Clamp: không vượt bounds nhưng giữ nguyên range (không stretch)
    if (newXMin < 0)  { newXMin = 0;  newXMax = xRange; }
    if (newXMax > xN) { newXMax = xN; newXMin = xN - xRange; }

    chart.options.scales.x.min = newXMin;
    chart.options.scales.x.max = newXMax;

    if (mode === 'xy') {
      const dy     = e.clientY - dragStart.y;
      const yRange = dragRange.yMax - dragRange.yMin;
      const yDelta = (dy / chartArea.height) * yRange;  // y inverted
      let newYMin  = dragRange.yMin + yDelta;
      let newYMax  = dragRange.yMax + yDelta;

      if (newYMin < 0) { newYMin = 0; newYMax = yRange; }
      if (newYMax > 1) { newYMax = 1; newYMin = 1 - yRange; }

      chart.options.scales.y.min = newYMin;
      chart.options.scales.y.max = newYMax;
    }

    chart.update('none');
  });

  const _stopDrag = () => {
    if (!isDragging) return;
    isDragging          = false;
    canvas.style.cursor = 'default';
  };
  canvas.addEventListener('mouseup',    _stopDrag);
  canvas.addEventListener('mouseleave', _stopDrag);

  // ── DOUBLE-CLICK RESET ─────────────────────────────────────────────────────
  canvas.addEventListener('dblclick', () => {
    chart.options.scales.x.min = undefined;
    chart.options.scales.x.max = undefined;
    if (mode === 'xy') {
      chart.options.scales.y.min = undefined;
      chart.options.scales.y.max = undefined;
    }
    chart.update('none');
  });

  canvas.style.cursor = 'default';
}

// ── Best epoch vertical line plugin (for loss curves) ────────────────────────
function _bestEpochPlugin(bestEpoch) {
  return {
    id: 'bestEpochLine',
    afterDraw(chart) {
      if (!bestEpoch) return;
      const ctx = chart.ctx;
      const xScale = chart.scales.x;
      if (!xScale) return;
      const xPos = xScale.getPixelForValue(bestEpoch - 1);
      const { top, bottom } = chart.chartArea;
      ctx.save();
      ctx.strokeStyle = '#e94560';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(xPos, top);
      ctx.lineTo(xPos, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#e94560';
      ctx.font = '10px IBM Plex Mono, monospace';
      ctx.fillText(`best:${bestEpoch}`, xPos + 4, top + 12);
      ctx.restore();
    },
  };
}

// =============================================================================
// 1. Predicted vs Actual (multi-model overlay, all 5 on one canvas)
// =============================================================================

/**
 * Build Predicted vs Actual line chart.
 * @param {string} canvasId
 * @param {Array<{label, dates, y_true, y_pred}>} seriesList
 */
function buildPredChart(canvasId, seriesList) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // y_true (Actual) — use first series
  const firstSeries = seriesList[0];
  // dates có thể null (regression không lưu dates trong npz)
  // → fallback dùng index 1..N làm X labels
  const labels = (firstSeries.dates && firstSeries.dates.length > 0)
    ? firstSeries.dates.map(d => String(d).slice(0, 10))
    : firstSeries.y_true.map((_, i) => i + 1);

  const datasets = [];

  // Actual line (once, shared)
  datasets.push({
    label: 'Actual',
    data: firstSeries.y_true,
    borderColor: '#cfd8dc', borderWidth: 1.5,
    pointRadius: 0, tension: 0.1,
  });

  // One predicted line per model
  seriesList.forEach(s => {
    const color = MODEL_COLORS[s.model] || '#64b5f6';
    datasets.push({
      label: `${s.model} (${s.wavelet ? 'W' : 'N'})`,
      data: s.y_pred,
      borderColor: color, borderWidth: 1.5,
      pointRadius: 0, tension: 0.1,
      borderDash: s.wavelet ? [] : [4, 3],
    });
  });

  const chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      ...BASE_OPTIONS,
      plugins: {
        ...BASE_OPTIONS.plugins,
        title: { display: true, text: 'Predicted vs Actual Close Price (VND)', color: '#e0e0e0', font: { size: 12 } },
      },
    },
  });
  _attachZoomPan(chart, 'x');
  _chartInstances[canvasId] = chart;
}

// =============================================================================
// 2. Training Loss Curves (single experiment)
// =============================================================================

/**
 * @param {string} canvasId
 * @param {{train_losses, val_losses, best_epoch, model_label}} data
 */
function buildLossChart(canvasId, data) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const labels = data.train_losses.map((_, i) => i + 1);
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Train Loss',
          data: data.train_losses,
          borderColor: '#64b5f6', borderWidth: 1.5, pointRadius: 0, tension: 0.2,
        },
        {
          label: 'Val Loss',
          data: data.val_losses,
          borderColor: '#e94560', borderWidth: 1.5, pointRadius: 0, tension: 0.2,
          borderDash: [5, 4],
        },
      ],
    },
    options: {
      ...BASE_OPTIONS,
      plugins: {
        ...BASE_OPTIONS.plugins,
        title: {
          display: true,
          text: data.model_label || 'Loss Curves',
          color: '#e0e0e0', font: { size: 11 },
        },
      },
      scales: {
        ...BASE_OPTIONS.scales,
        x: { ...BASE_OPTIONS.scales.x, title: { display: true, text: 'Epoch', color: '#9e9e9e', font: { size: 10 } } },
        y: { ...BASE_OPTIONS.scales.y, title: { display: true, text: 'Loss', color: '#9e9e9e', font: { size: 10 } } },
      },
    },
    // _bestEpochPlugin is an inline plugin (chart-level), separate from options.plugins
    plugins: [_bestEpochPlugin(data.best_epoch)],
  });
  _attachZoomPan(chart, 'x');
  _chartInstances[canvasId] = chart;
}

// =============================================================================
// 3. Trading Cumulative Return — Interactive Line Chart (NEW)
// =============================================================================

/**
 * Build interactive cumulative return chart.
 * @param {string} canvasId
 * @param {Array<{model, use_wavelet, dates, strategy, buyhold}>} seriesList
 * @param {boolean} showWavelet  — filter: true=wavelet, false=no-wavelet
 */
function buildTradingLineChart(canvasId, seriesList, showWavelet = true) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Filter by wavelet condition
  const filtered = seriesList.filter(s => s.use_wavelet === showWavelet);
  if (!filtered.length) return;

  // Use dates from first series
  const labels = filtered[0].dates.map(d => d.slice(0, 10));

  const datasets = [];

  // Buy & Hold (once — same for all models)
  datasets.push({
    label: 'Buy & Hold',
    data: filtered[0].buyhold.map(v => ((v - 1) * 100).toFixed(2)),
    borderColor: BUY_HOLD_COLOR, borderWidth: 2,
    pointRadius: 0, tension: 0.2, borderDash: [6, 4],
    fill: false,
  });

  // Strategy per model
  filtered.forEach(s => {
    const color = MODEL_COLORS[s.model] || '#64b5f6';
    datasets.push({
      label: s.model,
      data: s.strategy.map(v => ((v - 1) * 100).toFixed(2)),
      borderColor: color, borderWidth: 2,
      pointRadius: 0, tension: 0.2, fill: false,
    });
  });

  const chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      ...BASE_OPTIONS,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        ...BASE_OPTIONS.plugins,
        title: {
          display: true,
          text: `Cumulative Returns (Weekly) — ${showWavelet ? 'With Wavelet' : 'No Wavelet'} | Strategy vs Buy & Hold`,
          color: '#e0e0e0', font: { size: 12 },
        },
        tooltip: {
          ...BASE_OPTIONS.plugins.tooltip,
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` },
        },
      },
      scales: {
        x: {
          ...BASE_OPTIONS.scales.x,
          ticks: { ...BASE_OPTIONS.scales.x.ticks, maxTicksLimit: 12, maxRotation: 30 },
        },
        y: {
          ...BASE_OPTIONS.scales.y,
          title: { display: true, text: 'Cumulative Return (%)', color: '#9e9e9e', font: { size: 10 } },
        },
      },
    },
  });
  _attachZoomPan(chart, 'xy');
  _chartInstances[canvasId] = chart;
}

// =============================================================================
// 4. ROC Curves Overlay — 5 models on one chart (NEW)
// =============================================================================

/**
 * Build ROC overlay chart for up to 5 models.
 * @param {string} canvasId
 * @param {Array<{model, use_wavelet, fpr, tpr, auc_roc}>} rocList
 */
function buildRocOverlayChart(canvasId, rocList) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const datasets = [];

  // Diagonal reference line
  datasets.push({
    label: 'Random (AUC=0.50)',
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: '#546e7a', borderWidth: 1, borderDash: [5, 4],
    pointRadius: 0, fill: false,
    showLine: true,
  });

  // One curve per model
  rocList.forEach(r => {
    const color = MODEL_COLORS[r.model] || '#64b5f6';
    const wLabel = r.use_wavelet ? 'W' : 'N';
    datasets.push({
      label: `${r.model}(${wLabel}) AUC=${r.auc_roc.toFixed(3)}`,
      data: r.fpr.map((x, i) => ({ x, y: r.tpr[i] })),
      borderColor: color, borderWidth: 2,
      pointRadius: 0, tension: 0,
      fill: false,
      borderDash: r.use_wavelet ? [] : [4, 3],
    });
  });

  const chart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      ...BASE_OPTIONS,
      showLine: true,
      plugins: {
        ...BASE_OPTIONS.plugins,
        title: { display: true, text: 'ROC Curves — All Models (Fold selected)', color: '#e0e0e0', font: { size: 12 } },
      },
      scales: {
        x: {
          ...BASE_OPTIONS.scales.x, type: 'linear', min: 0, max: 1,
          title: { display: true, text: 'False Positive Rate', color: '#9e9e9e', font: { size: 10 } },
        },
        y: {
          ...BASE_OPTIONS.scales.y, type: 'linear', min: 0, max: 1,
          title: { display: true, text: 'True Positive Rate', color: '#9e9e9e', font: { size: 10 } },
        },
      },
    },
  });
  _attachZoomPan(chart, 'x');
  _chartInstances[canvasId] = chart;
}

// =============================================================================
// 5. Confusion Matrix HTML renderer
// =============================================================================

/**
 * Render a single confusion matrix as HTML inside a container div.
 * @param {string} containerId
 * @param {{model, stats: {TN,FP,FN,TP,total,accuracy}}} data
 */
function renderConfusionMatrix(containerId, data) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const { TN, FP, FN, TP, accuracy } = data.stats;
  el.innerHTML = `
    <div class="cm-model-name">${data.model}</div>
    <table class="cm-table">
      <thead>
        <tr>
          <th class="cm-header"></th>
          <th class="cm-header">Pred DOWN</th>
          <th class="cm-header">Pred UP</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="cm-header" style="font-size:9px">Act DOWN</td>
          <td class="cm-tn">${TN}</td>
          <td class="cm-fp">${FP}</td>
        </tr>
        <tr>
          <td class="cm-header" style="font-size:9px">Act UP</td>
          <td class="cm-fn">${FN}</td>
          <td class="cm-tp">${TP}</td>
        </tr>
      </tbody>
    </table>
    <div class="cm-acc">Acc: ${(accuracy * 100).toFixed(1)}%</div>`;
}

// =============================================================================
// Namespace export
// =============================================================================
window.VNSP = window.VNSP || {};
window.VNSP.charts = {
  buildPredChart,
  buildLossChart,
  buildTradingLineChart,
  buildRocOverlayChart,
  renderConfusionMatrix,
  destroyChart,
};