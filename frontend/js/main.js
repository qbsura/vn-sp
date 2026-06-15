/**
 * frontend/js/main.js
 * ====================
 * Dashboard controller for VNSP — Redesign 2026-06
 *
 * Sections: Data | Regression | Classification | Trading
 * All charts/plots support click-to-zoom (modal).
 *
 * Dependencies: Chart.js 4.4.2, api.js, charts.js
 * Namespace: window.VNSP.main
 */
'use strict';

// Shorthand refs (set after DOM ready)
let api, ui, charts;

// ── Constants ───────────────────────────────────────────────────────────────
const MODELS        = ['BiLSTM', 'LSTM', 'GRU', 'RNN', 'DNN'];
const TICKERS       = ['VCB', 'VIC'];
const CURRENCIES    = ['VND'];
const FOLDS         = [1, 2, 3];
const FOLD_LABELS   = { 1: 'Fold 1 (2018)', 2: 'Fold 2 (2020)', 3: 'Fold 3 (2022–24)' };

// Maps lowercase model name → canvas suffix id
const LOSS_CANVAS = { BiLSTM: 'bilstm', LSTM: 'lstm', GRU: 'gru', RNN: 'rnn', DNN: 'dnn' };

// ── Utilities ────────────────────────────────────────────────────────────────
const sel  = (id) => document.getElementById(id)?.value ?? '';
const setHTML = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };
const fmt  = (n, d = 4) => (n == null ? '—' : Number(n).toFixed(d));
const fmtPct = (n)      => (n == null ? '—' : `${(n * 100).toFixed(2)}%`);
const fmtVnd = (n)      => (n == null ? '—' : Number(n).toLocaleString('vi-VN'));

/** Build experiment ID */
function buildExpId(ticker, currency, useWavelet, model, task) {
  const cond = (useWavelet === true || useWavelet === 'true') ? 'wavelet' : 'nowave';
  return `${ticker}_${currency}_${cond}_${model}_${task}`;
}

/** Show/hide element */
const setVisible = (id, v) => document.getElementById(id)?.classList.toggle('hidden', !v);

// ── Image Zoom Modal ─────────────────────────────────────────────────────────
function initModal() {
  const overlay = document.getElementById('img-modal');
  const img     = document.getElementById('modal-img');
  const closeBtn = document.getElementById('modal-close-btn');

  // Make any img inside img-container zoomable
  document.addEventListener('click', (e) => {
    const target = e.target;
    if (target.tagName === 'IMG' && target.closest('.img-container, .cm-card, .plot-card-body')) {
      img.src = target.src;
      overlay.classList.add('visible');
    }
  });

  // Close on overlay click or close button
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.classList.remove('visible');
  });
  closeBtn.addEventListener('click', () => overlay.classList.remove('visible'));

  // Close on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') overlay.classList.remove('visible');
  });
}

/** Helper: set base64 image in img-container, enable zoom */
function setImg(containerId, base64Src, alt = '') {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = base64Src
    ? `<img src="${base64Src}" alt="${alt}" style="max-width:100%;border-radius:4px;" />`
    : `<div class="img-placeholder">No image data.</div>`;
}

// ── Navigation ───────────────────────────────────────────────────────────────
function switchSection(name) {
  document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(a => a.classList.remove('active'));
  document.getElementById(`section-${name}`)?.classList.add('active');
  document.querySelector(`.nav-item[data-section="${name}"]`)?.classList.add('active');
}

// ── API Status ────────────────────────────────────────────────────────────────
async function checkApiStatus() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (!dot) return;
  dot.className = 'status-dot checking'; text.textContent = 'Checking…';
  try {
    const res = await fetch('/health');
    if (res.ok) { dot.className = 'status-dot online'; text.textContent = 'API Online'; }
    else throw new Error(res.status);
  } catch {
    dot.className = 'status-dot offline'; text.textContent = 'API Offline';
  }
}

// =============================================================================
// SECTION 1 — DATA MANAGEMENT
// =============================================================================

/** Render a table body from rows array and column key list */
function renderTable(tbodyId, rows, colKeys) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  if (!rows?.length) {
    tbody.innerHTML = `<tr><td colspan="${colKeys.length}" class="table-empty">No data.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.slice(0, 10).map(row =>    // show 10 rows, rest via scroll
    `<tr>${colKeys.map(k => {
      const v = row[k];
      if (k === 'Date') return `<td>${String(v).slice(0, 10)}</td>`;
      if (k === 'Volume') return `<td>${Number(v).toLocaleString()}</td>`;
      return `<td>${v != null ? Number(v).toLocaleString('vi-VN') : '—'}</td>`;
    }).join('')}</tr>`
  ).join('');
}

async function loadData() {
  const ticker   = sel('data-ticker');
  const wavelet  = sel('data-wavelet') === 'true';
  const dateFrom = document.getElementById('data-date-from')?.value || '2012-01-01';
  const dateTo   = document.getElementById('data-date-to')?.value   || '2024-12-31';

  ui.showSpinner('Loading data…');
  try {
    // ── Load tables in parallel ──────────────────────────────────────────
    const [rawRes, featRes] = await Promise.allSettled([
      api.getRawData(ticker, 'VND', { start_date: dateFrom, end_date: dateTo, limit: 50 }),
      api.getFeatures(ticker, 'VND', wavelet, 50),
    ]);

    // Raw table — n_rows is total rows in dataset within date range, data is limited to 50
    if (rawRes.status === 'fulfilled' && rawRes.value?.data) {
      const nTotal = rawRes.value.n_rows ?? rawRes.value.data.length;
      document.getElementById('raw-count').textContent = `${nTotal} rows (showing 50)`;
      renderTable('raw-tbody', rawRes.value.data, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']);
    }

    // Features table (dynamic columns)
    if (featRes.status === 'fulfilled' && featRes.value?.sample) {
      const fv = featRes.value;
      document.getElementById('feat-count').textContent = `${fv.n_rows} rows · ${fv.feature_names.length} features`;
      const thead = document.getElementById('feat-thead');
      const cols  = fv.feature_names.slice(0, 8); // show max 8 cols
      if (thead) thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
      const rows = fv.sample.map(row =>
        `<tr>${cols.map(c => `<td>${row[c] != null ? Number(row[c]).toFixed(4) : '—'}</td>`).join('')}</tr>`
      );
      setHTML('feat-tbody', rows.join(''));
    }

    // ── Load plots sequentially (avoid hammering server) ──────────────────
    await _loadDataPlots(ticker, 'VND', wavelet);

  } catch (e) { /* toast shown by api.js */ }
  finally { ui.hideSpinner(); }
}

async function _loadDataPlots(ticker, currency, wavelet) {
  // ① Deviation Scatter (fig2 via viz endpoint)
  _loadVizImage('plot-deviation', () => api.getVizFig2(ticker, currency), 'Deviation Scatter');

  // ② Feature Distributions (fig3)
  _loadVizImage('plot-distributions', () => api.getVizFig3(ticker, currency), 'Distributions');

  // ③ db4 Wavelet & Scaling Functions (fig6, static)
  _loadVizImage('plot-wavelet-fn', () => api.getVizFig6(), 'Wavelet Functions');

  // ④ Wavelet Decomposition (fig5)
  _loadVizImage('plot-wavelet-decomp', () => api.getVizFig5(ticker, currency), 'Wavelet Decomp');

  // ⑤ Approx Coefficients (fig7)
  _loadVizImage('plot-approx', () => api.getVizFig7(ticker, currency), 'Approx Coeff');

  // ⑥ Detail Coefficients (fig8)
  _loadVizImage('plot-detail', () => api.getVizFig8(ticker, currency), 'Detail Coeff');

  // ⑦ Correlation Matrix (fig10)
  _loadVizImage('plot-correlation', () => api.getVizFig10(ticker, currency, wavelet), 'Correlation');
}

/** Generic viz image loader — shows placeholder while loading */
async function _loadVizImage(containerId, fetchFn, label) {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `<div class="img-placeholder">Loading ${label}…</div>`;
  try {
    const res = await fetchFn();
    setImg(containerId, res.image, label);
  } catch {
    if (el) el.innerHTML = `<div class="img-placeholder text-red">Failed to load ${label}</div>`;
  }
}


/** Render architecture visualization as HTML */



// =============================================================================
// SECTION 3 — REGRESSION
// =============================================================================

async function loadRegression() {
  const ticker  = sel('reg-ticker');
  const wavelet = sel('reg-wavelet') === 'true';   // controls pred chart, loss curves, primary table
  const fold    = parseInt(sel('reg-fold'), 10) || 3;
  const wLabel  = wavelet ? 'With Wavelet' : 'No Wavelet';

  // Update primary table label
  const labelEl = document.getElementById('reg-table-label');
  if (labelEl) labelEl.textContent = `${ticker} · VND · ${wLabel} · ${FOLD_LABELS[fold]}`;

  ui.showSpinner('Loading regression results…');
  try {
    // ── Fetch comparison table (both conditions in one call) ──────────────
    const tbl = await api.getComparisonTable(ticker, 'VND', 'regression').catch(() => null);
    if (tbl) _renderRegressionTable(tbl, wavelet);  // renders primary + Before/After

    // ── Predicted vs Actual — all 5 models (selected wavelet condition) ───
    const predPromises = MODELS.map(model => {
      const expId = buildExpId(ticker, 'VND', wavelet, model, 'regression');
      return api.getPredictions(expId, fold)
        .then(r => ({
          model, wavelet,
          y_true: Array.isArray(r.y_true) ? r.y_true : [],
          y_pred: Array.isArray(r.y_pred) ? r.y_pred : [],
          dates : Array.isArray(r.dates) && r.dates.length ? r.dates : null,
        }))
        .catch(() => null);
    });
    const predResults = (await Promise.allSettled(predPromises))
      .filter(r => r.status === 'fulfilled' && r.value?.y_true?.length)
      .map(r => r.value);

    if (predResults.length > 0) charts.buildPredChart('pred-chart', predResults);

    // ── Loss curves — one canvas per model ───────────────────────────────
    MODELS.forEach(async model => {
      const expId = buildExpId(ticker, 'VND', wavelet, model, 'regression');
      try {
        const lc = await api.getLossCurves(expId, fold);
        charts.buildLossChart(`loss-chart-${LOSS_CANVAS[model]}`, {
          train_losses: lc.train_losses,
          val_losses  : lc.val_losses,
          best_epoch  : lc.best_epoch,
          model_label : model,
        });
      } catch { /* skip if no data */ }
    });

  } catch (e) { /* toast shown by api.js */ }
  finally { ui.hideSpinner(); }
}

/**
 * Render regression metrics tables.
 * Fills 3 table bodies:
 *   1. reg-metrics-tbody  — primary: 5 models for the SELECTED wavelet condition
 *   2. reg-tbody-before   — comparison: No Wavelet (before)
 *   3. reg-tbody-after    — comparison: With Wavelet (after)
 *
 * RMSE = √MSE (computed client-side — not stored in old metrics.json).
 * R² is NOT shown (not available in stored experiments).
 */
function _renderRegressionTable(tbl, showWavelet) {
  const models = tbl.models || MODELS;

  // 1. Primary table: selected condition
  const primaryData = showWavelet ? tbl.after_wavelet : tbl.before_wavelet;
  _renderRegressionHalf('reg-metrics-tbody', models, primaryData);

  // 2 + 3. Before/After comparison tables (always both)
  // Before/After table: 4 metrics only (no R²: API thường null, gây cột trống)
  const WAVE_METRICS = ['MSE', 'MAE', 'MAPE', 'RMSE'];
  _renderRegressionHalf('reg-tbody-before', models, tbl.before_wavelet, WAVE_METRICS);
  _renderRegressionHalf('reg-tbody-after',  models, tbl.after_wavelet,  WAVE_METRICS);
}

/**
 * Fill one <tbody> with 5-model metrics for a single wavelet condition.
 * @param {string}   tbodyId  DOM id of <tbody> element
 * @param {string[]} models   Model name list (e.g. ['BiLSTM','LSTM',…])
 * @param {Object}   data     API object: {MSE:[…], MAE:[…], MAPE:[…], …}
 */
function _renderRegressionHalf(tbodyId, models, data,
  /* optional: subset of metrics to show */
  metrics = ['MSE', 'MAE', 'MAPE', 'RMSE', 'R2']) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;

  if (!data) {
    tbody.innerHTML = `<tr><td colspan="${metrics.length + 1}" class="table-empty">No data.</td></tr>`;
    return;
  }

  // Compute RMSE = √MSE client-side (not stored in old metrics.json)
  const mseArr  = data.MSE || [];
  const rmseArr = mseArr.map(v => (v != null ? Math.sqrt(v) : null));
  const enriched = { ...data, RMSE: rmseArr };

  // metrics: được truyền từ caller (primary table = 5 cols incl. R²;
  //          before/after table = 4 cols, không có R² để tránh cột trống)

  // Find best per metric: min for MSE/MAE/MAPE/RMSE; max for R²
  const best = {};
  metrics.forEach(m => {
    const vals = (enriched[m] || []).filter(v => v != null && isFinite(v));
    if (!vals.length) return;
    best[m] = m === 'R2' ? Math.max(...vals) : Math.min(...vals);
  });

  tbody.innerHTML = models.map((model, i) => {
    const cells = metrics.map(m => {
      const v = enriched[m]?.[i];
      if (v == null || !isFinite(v)) return '<td>—</td>';
      const isBest = best[m] != null && Math.abs(v - best[m]) < 1e-10;
      return `<td class="${isBest ? 'best-val' : ''}">${fmt(v)}</td>`;
    }).join('');
    return `<tr><td class="model-name">${model}</td>${cells}</tr>`;
  }).join('');
}

// =============================================================================
// SECTION 4 — CLASSIFICATION
// =============================================================================

async function loadClassification() {
  const ticker  = sel('cls-ticker');
  const wavelet = sel('cls-wavelet') === 'true';
  const fold    = parseInt(sel('cls-fold'), 10) || 3;

  ui.showSpinner('Loading classification results…');
  try {
    // ── Fetch classification report for all 5 models in parallel ─────────
    // API scans all 3 folds internally — không cần truyền fold_idx
    // threshold default 0.5 (không cần truyền)
    const reportPromises = MODELS.map(model => {
      const expId = buildExpId(ticker, 'VND', wavelet, model, 'classification');
      return api.getClassificationReport(expId)
        .then(r => {
          // Lấy metrics của fold được chọn, fallback sang aggregated
          const foldData = r.folds?.find(f => f.fold_idx === fold);
          const metrics  = foldData?.metrics || {};
          return { model, metrics, aggregated: r.aggregated, error: false };
        })
        .catch(() => ({ model, metrics: {}, aggregated: null, error: true }));
    });
    const reports = await Promise.all(reportPromises);

    // ── Metrics table ─────────────────────────────────────────────────────
    _renderClassificationTable(reports);

    // ── Confusion matrices ────────────────────────────────────────────────
    const cmPromises = MODELS.map(model => {
      const expId = buildExpId(ticker, 'VND', wavelet, model, 'classification');
      return api.getConfusionMatrix(expId, fold)
        .then(r => ({ model, stats: r.stats }))
        .catch(() => null);
    });
    const cmResults = await Promise.all(cmPromises);
    cmResults.forEach(cm => {
      if (!cm) return;
      charts.renderConfusionMatrix(`cm-${cm.model}`, cm);
    });

    // ── ROC curves overlay ────────────────────────────────────────────────
    const rocPromises = MODELS.map(model => {
      const expId = buildExpId(ticker, 'VND', wavelet, model, 'classification');
      return api.getROCCurve(expId, fold)
        .then(r => ({ model, use_wavelet: wavelet, fpr: r.fpr, tpr: r.tpr, auc_roc: r.auc_roc }))
        .catch(() => null);
    });
    const rocResults = (await Promise.all(rocPromises)).filter(Boolean);
    if (rocResults.length) charts.buildRocOverlayChart('roc-chart', rocResults);

  } catch (e) { /* toast */ }
  finally { ui.hideSpinner(); }
}

function _renderClassificationTable(reports) {
  const tbody = document.getElementById('cls-metrics-tbody');
  if (!tbody) return;

  // compute_classification_metrics returns PascalCase: Accuracy, Precision, Recall, F1, AUC_ROC
  const metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC'];

  // Find best per metric across models
  const best = {};
  metrics.forEach(m => {
    const vals = reports.map(r => r.metrics?.[m]).filter(v => v != null && !isNaN(v));
    if (vals.length) best[m] = Math.max(...vals);
  });

  tbody.innerHTML = reports.map(r => {
    if (r.error) {
      return `<tr><td class="model-name">${r.model}</td><td colspan="5" class="text-red">No data</td></tr>`;
    }
    const m_data = r.metrics || {};
    const cells = metrics.map(m => {
      const v = m_data[m];
      if (v == null || isNaN(v)) return '<td>—</td>';
      const isBest = best[m] != null && Math.abs(v - best[m]) < 1e-6;
      return `<td class="${isBest ? 'best-val' : ''}">${fmtPct(v)}</td>`;
    }).join('');
    return `<tr><td class="model-name">${r.model}</td>${cells}</tr>`;
  }).join('');
}

// =============================================================================
// SECTION 5 — TRADING SIMULATION
// =============================================================================

let _tradeWavelet = true; // current tab selection

async function loadTrading() {
  const ticker = sel('trade-ticker');
  const fold   = parseInt(sel('trade-fold'), 10) || 3;

  ui.showSpinner('Loading trading simulation…');
  try {
    // ── Fetch timeseries + summary in parallel ─────────────────────────────
    const [tsRes, sumRes] = await Promise.allSettled([
      api.getTradingTimeseries(ticker, 'VND', fold),
      api.getTradingResults(ticker, 'VND', fold),
    ]);

    // Chart.js line chart — cache data on canvas element for tab switching
    if (tsRes.status === 'fulfilled' && tsRes.value?.data?.length) {
      const canvas = document.getElementById('trade-chart');
      if (canvas) canvas._tsData = tsRes.value.data;  // save for _switchTradeTab()
      charts.buildTradingLineChart('trade-chart', tsRes.value.data, _tradeWavelet);
    }

    // Summary table
    if (sumRes.status === 'fulfilled' && sumRes.value?.data?.length) {
      _renderTradingTable(sumRes.value.data);
    }

  } catch (e) { /* toast */ }
  finally { ui.hideSpinner(); }
}

function _renderTradingTable(rows) {
  const tbody = document.getElementById('trade-tbody');
  if (!tbody) return;

  const validRows = rows.filter(r => r.status === 'ok' || r.Cumulative_Return != null);

  // Find best cumulative return per wavelet group
  const waveRows  = validRows.filter(r => r.wavelet === true);
  const noWave    = validRows.filter(r => r.wavelet === false);
  const bestCumW  = Math.max(...waveRows.map(r => r.Cumulative_Return || 0));
  const bestCumN  = Math.max(...noWave.map(r => r.Cumulative_Return || 0));

  tbody.innerHTML = rows.map(r => {
    const waveLabel = r.wavelet ? '<span class="text-blue">✓ Wave</span>' : 'No Wave';
    const cum = r.Cumulative_Return;
    const bnh = r.BuyHold_Return;
    const isBest = r.wavelet ? Math.abs(cum - bestCumW) < 1e-4 : Math.abs(cum - bestCumN) < 1e-4;

    return `<tr>
      <td class="model-name">${r.model}</td>
      <td>${waveLabel}</td>
      <td class="${isBest ? 'best-val' : ''}">${fmtPct(cum)}</td>
      <td>${fmtPct(bnh)}</td>
      <td>${r.Sharpe_Ratio != null ? fmt(r.Sharpe_Ratio, 3) : '—'}</td>
      <td>${r.Max_Drawdown != null ? fmtPct(r.Max_Drawdown) : '—'}</td>
      <td>${r.Win_Rate != null ? fmtPct(r.Win_Rate) : '—'}</td>
    </tr>`;
  }).join('');
}

// Switch wavelet tab → redraw chart
function _switchTradeTab(showWavelet) {
  _tradeWavelet = showWavelet;
  document.getElementById('trade-tab-wave')?.classList.toggle('active', showWavelet);
  document.getElementById('trade-tab-nowave')?.classList.toggle('active', !showWavelet);
  // Re-render chart if data already loaded (stored in chart)
  const canvas = document.getElementById('trade-chart');
  if (canvas._tsData) charts.buildTradingLineChart('trade-chart', canvas._tsData, showWavelet);
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function attachEventListeners() {
  const on = (id, fn) => document.getElementById(id)?.addEventListener('click', fn);

  // Navigation
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      switchSection(item.dataset.section);
    });
  });

  // Section 1: Data
  on('btn-load-data', loadData);

  // Section 3: Regression
  on('btn-load-regression', loadRegression);

  // Section 4: Classification
  on('btn-load-classification', loadClassification);

  // Section 5: Trading
  on('btn-load-trading', loadTrading);

  // Trading wavelet tabs
  document.getElementById('trade-tab-wave')  ?.addEventListener('click', () => _switchTradeTab(true));
  document.getElementById('trade-tab-nowave')?.addEventListener('click', () => _switchTradeTab(false));
}

// =============================================================================
// INIT
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
  api    = window.VNSP.api;
  ui     = window.VNSP.ui;
  charts = window.VNSP.charts;

  initModal();
  attachEventListeners();
  checkApiStatus();

  // Auto-load data section on startup
  loadData().catch(() => {});
});

// Export public methods
window.VNSP = window.VNSP || {};
window.VNSP.main = { switchSection, loadData, loadRegression, loadClassification, loadTrading };