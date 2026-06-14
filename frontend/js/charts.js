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

// ── Wheel zoom + drag-pan (inline plugin — no CDN needed) ────────────────────
const _zoomPanPlugin = {
  id: 'zoomPan',
  afterInit(chart) {
    const canvas = chart.canvas;
    let isDragging = false, dragStartX = 0, dragStartY = 0;
    let origMin = {}, origMax = {};

    // Save original axis bounds
    function _saveOrig() {
      for (const id in chart.scales) {
        const s = chart.scales[id];
        origMin[id] = s.min ?? s._range?.min;
        origMax[id] = s._range?.max;
      }
    }

    // Wheel zoom
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.15 : 0.87;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;

      for (const id in chart.scales) {
        const s = chart.scales[id];
        if (s.type === 'category') continue;
        const min = s.min ?? s.min, max = s.max ?? s.max;
        const ratio = id.startsWith('x')
          ? (s.getValueForPixel(px) - min) / (max - min)
          : (s.getValueForPixel(py) - min) / (max - min);
        const range = (max - min) * factor;
        const center = s.getValueForPixel(id.startsWith('x') ? px : py);
        s.options.min = center - range * ratio;
        s.options.max = center + range * (1 - ratio);
      }
      chart.update('none');
    }, { passive: false });

    // Drag pan
    canvas.addEventListener('mousedown', (e) => {
      isDragging = true; dragStartX = e.clientX; dragStartY = e.clientY;
    });
    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - dragStartX, dy = e.clientY - dragStartY;
      dragStartX = e.clientX; dragStartY = e.clientY;
      for (const id in chart.scales) {
        const s = chart.scales[id];
        if (s.type === 'category' || s.min == null) continue;
        const range = (s.max - s.min);
        const delta = id.startsWith('x')
          ? -dx / canvas.width * range
          : dy / canvas.height * range;
        s.options.min = s.min + delta;
        s.options.max = s.max + delta;
      }
      chart.update('none');
    });
    window.addEventListener('mouseup', () => { isDragging = false; });

    // Double-click: reset zoom
    canvas.addEventListener('dblclick', () => {
      for (const id in chart.scales) {
        const s = chart.scales[id];
        s.options.min = undefined;
        s.options.max = undefined;
      }
      chart.update();
    });
  },
};

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
    plugins: [_zoomPanPlugin],
  });
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
    plugins: [_zoomPanPlugin, _bestEpochPlugin(data.best_epoch)],
  });
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
    plugins: [_zoomPanPlugin],
  });
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
    plugins: [_zoomPanPlugin],
  });
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