/**
 * frontend/js/api.js
 * ==================
 * API client for VNSP Dashboard — Task 8.3
 *
 * All functions are async and return parsed JSON.
 * Errors are caught, displayed as toast, and re-thrown so callers can
 * hide spinners / reset UI in their own catch blocks.
 *
 * UI helpers (toast, spinner) live here — shared with main.js via
 * window.VNSP.ui namespace.
 *
 * Polling: pollJob(jobId, onProgress, onDone, onError) — 2 s interval.
 *
 * Namespace: window.VNSP.api
 */

'use strict';

const API_BASE = '/api';

// =============================================================================
// UI HELPERS — toast & spinner
// =============================================================================

/** Active polling timers: jobId → intervalId */
const _polls = {};

/**
 * Show a toast notification.
 * @param {'success'|'error'|'warning'|'info'} type
 * @param {string} title
 * @param {string} [msg]
 * @param {number} [duration]  ms before auto-dismiss (0 = manual only)
 */
function showToast(type, title, msg = '', duration = 4500) {
  const container = document.getElementById('toast-container');
  if (!container) { console.warn('[toast]', title, msg); return; }

  const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
  const toast  = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] ?? '•'}</span>
    <div class="toast-body">
      <div class="toast-title">${title}</div>
      ${msg ? `<div class="toast-msg">${msg}</div>` : ''}
    </div>
    <button class="toast-close" aria-label="Dismiss">✕</button>`;

  toast.querySelector('.toast-close').addEventListener('click', () => _dismissToast(toast));
  container.appendChild(toast);

  if (duration > 0) {
    setTimeout(() => _dismissToast(toast), duration);
  }
}

function _dismissToast(toast) {
  toast.classList.add('toast-exit');
  setTimeout(() => toast.remove(), 280);
}

/**
 * Show full-screen loading spinner.
 * @param {string} [text]
 */
function showSpinner(text = 'Loading…') {
  const el = document.getElementById('spinner-overlay');
  const tx = document.getElementById('spinner-text');
  if (el) { el.classList.remove('hidden'); }
  if (tx)   tx.textContent = text;
}

/** Hide full-screen loading spinner. */
function hideSpinner() {
  const el = document.getElementById('spinner-overlay');
  if (el) el.classList.add('hidden');
}


// =============================================================================
// CORE FETCH WRAPPER
// =============================================================================

/**
 * Internal fetch with error handling.
 * Throws on HTTP errors (4xx / 5xx) after showing a toast.
 *
 * @param {string}  url
 * @param {object}  [options]  fetch init options
 * @returns {Promise<any>}     parsed JSON response
 */
async function _fetch(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail ?? err.message ?? detail;
    } catch (_) { /* ignore JSON parse error */ }
    showToast('error', `API Error ${res.status}`, String(detail).slice(0, 120));
    throw new Error(`${url} → ${res.status}: ${detail}`);
  }

  return res.json();
}

/** Build query string from a plain object (skips undefined/null). */
function _qs(params) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') p.append(k, v);
  }
  const s = p.toString();
  return s ? `?${s}` : '';
}


// =============================================================================
// DATA API  — /api/data/*
// =============================================================================

/** GET /api/data/status */
async function checkDataStatus() {
  return _fetch(`${API_BASE}/data/status`);
}

/**
 * GET /api/data/{ticker}/raw
 * @param {string} ticker
 * @param {string} currency
 * @param {object} [params]  { start_date, end_date, limit }
 */
async function getRawData(ticker, currency = 'VND', params = {}) {
  return _fetch(`${API_BASE}/data/${ticker}/raw${_qs({ currency, ...params })}`);
}

/**
 * GET /api/data/{ticker}/features
 * @param {string}  ticker
 * @param {string}  currency
 * @param {boolean} useWavelet
 * @param {number}  [limit]
 */
async function getFeatures(ticker, currency = 'VND', useWavelet = true, limit = 50) {
  return _fetch(`${API_BASE}/data/${ticker}/features${_qs({ currency, use_wavelet: useWavelet, limit })}`);
}

/**
 * GET /api/data/{ticker}/deviation-plot
 * @returns {Promise<{image: string, ticker: string, currency: string}>}
 */
async function getDeviationPlot(ticker, currency = 'VND') {
  return _fetch(`${API_BASE}/data/${ticker}/deviation-plot${_qs({ currency })}`);
}

/**
 * POST /api/data/preprocess  (query-param style per FastAPI convention)
 * @returns {Promise<{status, n_features, n_rows}>}
 */
async function triggerPreprocess(ticker, currency = 'VND', useWavelet = true) {
  return _fetch(`${API_BASE}/data/preprocess${_qs({ ticker, currency, use_wavelet: useWavelet })}`, {
    method: 'POST',
  });
}


// =============================================================================
// EXPERIMENTS API  — /api/experiments/*
// =============================================================================

/**
 * GET /api/experiments/matrix
 * @param {object} [filters]  { ticker, currency, task, wavelet }
 */
async function getExperimentMatrix(filters = {}) {
  return _fetch(`${API_BASE}/experiments/matrix${_qs(filters)}`);
}

/**
 * POST /api/experiments/run  (returns job_id immediately)
 * @param {object} config  { ticker, currency, use_wavelet, model, task, fold? }
 * @returns {Promise<{job_id, status, exp_id}>}
 */
async function runExperiment(config) {
  return _fetch(`${API_BASE}/experiments/run`, {
    method : 'POST',
    body   : JSON.stringify(config),
  });
}

/**
 * GET /api/experiments/{jobId}/status
 * @returns {Promise<{job_id, status, type, progress, error, exp_id}>}
 */
async function getJobStatus(jobId) {
  return _fetch(`${API_BASE}/experiments/${jobId}/status`);
}

/**
 * POST /api/experiments/hpo
 * @param {object} config  { ticker, currency, use_wavelet, n_trials }
 * @returns {Promise<{job_id, status, label, n_trials}>}
 */
async function runHPO(config) {
  return _fetch(`${API_BASE}/experiments/hpo`, {
    method : 'POST',
    body   : JSON.stringify(config),
  });
}

/**
 * GET /api/experiments/{expId}/params
 * @param {string} expId
 * @param {number} [foldIdx]
 */
async function getBestParams(expId, foldIdx = 1) {
  return _fetch(`${API_BASE}/experiments/${expId}/params${_qs({ fold_idx: foldIdx })}`);
}

/**
 * GET /api/experiments/{expId}/loss-curves
 * @param {string} expId
 * @param {number} [foldIdx]
 * @returns {Promise<{exp_id, fold_idx, train_losses, val_losses, best_epoch, stopped_early}>}
 */
async function getLossCurves(expId, foldIdx = 1) {
  return _fetch(`${API_BASE}/experiments/${expId}/loss-curves${_qs({ fold_idx: foldIdx })}`);
}


// =============================================================================
// RESULTS API  — /api/results/*
// =============================================================================

/**
 * GET /api/results/summary
 * @param {object} [filters]  { ticker, currency, task, wavelet }
 */
async function getResultsSummary(filters = {}) {
  return _fetch(`${API_BASE}/results/summary${_qs(filters)}`);
}

/**
 * GET /api/results/comparison-table  (replica Table 1 bài báo)
 * @param {string} ticker
 * @param {string} currency
 * @param {string} [task]
 */
async function getComparisonTable(ticker, currency = 'VND', task = 'regression') {
  return _fetch(`${API_BASE}/results/comparison-table${_qs({ ticker, currency, task })}`);
}

/** GET /api/results/best-models */
async function getBestModels(ticker = null, currency = null) {
  return _fetch(`${API_BASE}/results/best-models${_qs({ ticker, currency })}`);
}

/**
 * GET /api/results/{expId}/predictions
 * @param {string} expId
 * @param {number} [foldIdx]
 */
async function getPredictions(expId, foldIdx = 3) {
  return _fetch(`${API_BASE}/results/${expId}/predictions${_qs({ fold_idx: foldIdx })}`);
}

/**
 * GET /api/results/{expId}/classification-report
 * @param {string} expId  must contain "classification"
 * @param {number} [threshold]
 */
async function getClassificationReport(expId, threshold = 0.5) {
  return _fetch(`${API_BASE}/results/${expId}/classification-report${_qs({ threshold })}`);
}

/**
 * GET /api/results/{expId}/confusion-matrix
 * @param {string} expId
 * @param {number} [foldIdx]
 * @param {number} [threshold]
 */
async function getConfusionMatrix(expId, foldIdx = 3, threshold = 0.5) {
  return _fetch(`${API_BASE}/results/${expId}/confusion-matrix${_qs({ fold_idx: foldIdx, threshold })}`);
}

/**
 * GET /api/results/{expId}/roc-curve
 * @param {string} expId
 * @param {number} [foldIdx]
 */
async function getROCCurve(expId, foldIdx = 3) {
  return _fetch(`${API_BASE}/results/${expId}/roc-curve${_qs({ fold_idx: foldIdx })}`);
}

/**
 * GET /api/results/trading/{ticker}/{currency}
 * @param {string} ticker
 * @param {string} currency
 * @param {number} [foldIdx]
 */
async function getTradingResults(ticker, currency = 'VND', foldIdx = 3) {
  return _fetch(`${API_BASE}/results/trading/${ticker}/${currency}${_qs({ fold_idx: foldIdx })}`);
}


// =============================================================================
// VIZ API  — /api/viz/*
// Tất cả endpoints trả về: { image: "data:image/png;base64,..." }
// Dùng cho Section Figures (matplotlib server-side rendering).
// =============================================================================

// ── Paper figures (Fig. 1–11) ────────────────────────────────────────────────

/** GET /api/viz/fig1 — Pipeline framework diagram (static, no params) */
async function getVizFig1() {
  return _fetch(`${API_BASE}/viz/fig1`);
}

/**
 * GET /api/viz/fig2 — Deviation scatter plot
 * @param {string} ticker   'VCB' | 'VIC'
 * @param {string} currency 'VND' | 'USD'
 */
async function getVizFig2(ticker = 'VCB', currency = 'VND') {
  return _fetch(`${API_BASE}/viz/fig2${_qs({ ticker, currency })}`);
}

/**
 * GET /api/viz/fig3 — Wavelet feature distribution histograms
 * @param {string} ticker
 * @param {string} currency
 */
async function getVizFig3(ticker = 'VCB', currency = 'VND') {
  return _fetch(`${API_BASE}/viz/fig3${_qs({ ticker, currency })}`);
}

/** GET /api/viz/fig4 — Feature scaling flowchart (static, no params) */
async function getVizFig4() {
  return _fetch(`${API_BASE}/viz/fig4`);
}

/**
 * GET /api/viz/fig5 — Wavelet decomposition time series
 * @param {string} ticker
 * @param {string} currency
 */
async function getVizFig5(ticker = 'VCB', currency = 'VND') {
  return _fetch(`${API_BASE}/viz/fig5${_qs({ ticker, currency })}`);
}

/** GET /api/viz/fig6 — db4 wavelet & scaling functions (static, no params) */
async function getVizFig6() {
  return _fetch(`${API_BASE}/viz/fig6`);
}

/**
 * GET /api/viz/fig7 — Approximation coefficients (A1) — VIC analog
 * @param {string} ticker   default 'VIC' (Tesla analog trong bài báo)
 * @param {string} currency
 */
async function getVizFig7(ticker = 'VIC', currency = 'VND') {
  return _fetch(`${API_BASE}/viz/fig7${_qs({ ticker, currency })}`);
}

/**
 * GET /api/viz/fig8 — Detail coefficients (D1) — VCB analog
 * @param {string} ticker   default 'VCB' (Apple analog trong bài báo)
 * @param {string} currency
 */
async function getVizFig8(ticker = 'VCB', currency = 'VND') {
  return _fetch(`${API_BASE}/viz/fig8${_qs({ ticker, currency })}`);
}

/** GET /api/viz/fig9 — Level-1 SWT decomposition diagram (static, no params) */
async function getVizFig9() {
  return _fetch(`${API_BASE}/viz/fig9`);
}

/**
 * GET /api/viz/fig10 — Feature correlation matrix heatmap
 * @param {string}  ticker
 * @param {string}  currency
 * @param {boolean} wavelet  true = wavelet features, false = raw features
 */
async function getVizFig10(ticker = 'VCB', currency = 'VND', wavelet = true) {
  return _fetch(`${API_BASE}/viz/fig10${_qs({ ticker, currency, wavelet })}`);
}

/**
 * GET /api/viz/fig11 — MSE comparison bar chart (replica Fig. 11 bài báo)
 * @param {string} ticker
 * @param {string} currency
 */
async function getVizFig11(ticker = 'VCB', currency = 'VND') {
  return _fetch(`${API_BASE}/viz/fig11${_qs({ ticker, currency })}`);
}


// ── Extended figures ─────────────────────────────────────────────────────────

/**
 * GET /api/viz/predicted-vs-actual — Predicted vs Actual price chart
 * @param {string} expId     Experiment ID (regression task)
 * @param {number} [foldIdx] 1 | 2 | 3
 */
async function getVizPredVsActual(expId, foldIdx = 3) {
  return _fetch(`${API_BASE}/viz/predicted-vs-actual${_qs({ exp_id: expId, fold_idx: foldIdx })}`);
}

/**
 * GET /api/viz/loss-curves — Train/val loss curves per epoch (matplotlib)
 * @param {string} expId
 * @param {number} [foldIdx]
 */
async function getVizLossCurves(expId, foldIdx = 1) {
  return _fetch(`${API_BASE}/viz/loss-curves${_qs({ exp_id: expId, fold_idx: foldIdx })}`);
}

/**
 * GET /api/viz/confusion-matrix — Confusion matrix heatmap (matplotlib/seaborn)
 * @param {string} expId     Experiment ID (classification task)
 * @param {number} [foldIdx]
 */
async function getVizConfusionMatrix(expId, foldIdx = 3) {
  return _fetch(`${API_BASE}/viz/confusion-matrix${_qs({ exp_id: expId, fold_idx: foldIdx })}`);
}

/**
 * GET /api/viz/trading-returns — Cumulative return chart (5 models vs Buy & Hold)
 * @param {string} ticker
 * @param {string} currency
 * @param {number} [foldIdx]
 */
async function getVizTradingReturns(ticker = 'VCB', currency = 'VND', foldIdx = 3) {
  return _fetch(`${API_BASE}/viz/trading-returns${_qs({ ticker, currency, fold_idx: foldIdx })}`);
}

/**
 * GET /api/viz/walkforward — Walk-forward stability chart (metric per fold, 5 models)
 * @param {string}  ticker
 * @param {string}  currency
 * @param {string}  [task]    'regression' | 'classification'
 * @param {string}  [metric]  'MSE' | 'MAE' | 'MAPE' | 'R2' | 'Accuracy' | 'F1' | 'AUC_ROC'
 * @param {boolean} [wavelet]
 */
async function getVizWalkforward(
  ticker   = 'VCB',
  currency = 'VND',
  task     = 'regression',
  metric   = 'MSE',
  wavelet  = true,
) {
  return _fetch(`${API_BASE}/viz/walkforward${_qs({ ticker, currency, task, metric, wavelet })}`);
}

/**
 * GET /api/viz/roc-curves — Multi-model ROC curves overlay (5 models)
 * @param {string}  ticker
 * @param {string}  currency
 * @param {number}  [foldIdx]
 * @param {boolean} [wavelet]
 */
async function getVizRocCurves(ticker = 'VCB', currency = 'VND', foldIdx = 3, wavelet = true) {
  return _fetch(`${API_BASE}/viz/roc-curves${_qs({ ticker, currency, fold_idx: foldIdx, wavelet })}`);
}

/**
 * GET /api/viz/fig-classification-table — Classification metrics table (Accuracy/F1/AUC per model per fold)
 * @param {string}  ticker
 * @param {string}  currency
 * @param {boolean} [wavelet]
 */
async function getVizFigClassificationTable(ticker = 'VCB', currency = 'VND', wavelet = true) {
  return _fetch(`${API_BASE}/viz/fig-classification-table${_qs({ ticker, currency, wavelet })}`);
}

/**
 * GET /api/viz/fig-walkforward — Walk-forward stability chart cho một experiment cụ thể
 * @param {string} expId  Experiment ID đầy đủ (e.g. 'VCB_VND_wavelet_BiLSTM_regression')
 */
async function getVizFigWalkforward(expId) {
  return _fetch(`${API_BASE}/viz/fig-walkforward${_qs({ exp_id: expId })}`);
}

/**
 * GET /api/viz/fig-cumulative-return — Cumulative return chart với Sharpe/MaxDD annotations
 * @param {string}  ticker
 * @param {string}  currency
 * @param {number}  [foldIdx]
 * @param {boolean} [wavelet]
 */
async function getVizFigCumulativeReturn(ticker = 'VCB', currency = 'VND', foldIdx = 3, wavelet = true) {
  return _fetch(`${API_BASE}/viz/fig-cumulative-return${_qs({ ticker, currency, fold_idx: foldIdx, wavelet })}`);
}


// =============================================================================
// POLLING — long-running experiments / HPO
// =============================================================================

/**
 * Poll job status every intervalMs until done or error.
 * Automatically stops and cleans up on terminal state.
 *
 * @param {string}   jobId
 * @param {Function} onProgress  (statusObj) => void — called on each tick
 * @param {Function} onDone      (statusObj) => void — called when status === 'done'
 * @param {Function} onError     (errMsg: string) => void
 * @param {number}   [intervalMs]  default 2000
 * @returns {Function}  cancel() — call to stop polling early
 */
function pollJob(jobId, onProgress, onDone, onError, intervalMs = 2000) {
  // Clear any existing poll for same jobId
  if (_polls[jobId]) clearInterval(_polls[jobId]);

  const timer = setInterval(async () => {
    try {
      const status = await getJobStatus(jobId);
      onProgress(status);

      if (status.status === 'done') {
        clearInterval(timer);
        delete _polls[jobId];
        onDone(status);
      } else if (status.status === 'error') {
        clearInterval(timer);
        delete _polls[jobId];
        onError(status.error ?? 'Unknown error');
      }
    } catch (err) {
      clearInterval(timer);
      delete _polls[jobId];
      onError(err.message);
    }
  }, intervalMs);

  _polls[jobId] = timer;

  // Return cancel function
  return () => {
    clearInterval(timer);
    delete _polls[jobId];
  };
}


// =============================================================================
// NAMESPACE EXPORT
// =============================================================================

window.VNSP = window.VNSP || {};
window.VNSP.ui  = { showToast, showSpinner, hideSpinner };
window.VNSP.api = {
  // Data
  checkDataStatus,
  getRawData,
  getFeatures,
  getDeviationPlot,
  triggerPreprocess,
  // Experiments
  getExperimentMatrix,
  runExperiment,
  getJobStatus,
  runHPO,
  getBestParams,
  getLossCurves,
  // Results
  getResultsSummary,
  getComparisonTable,
  getBestModels,
  getPredictions,
  getClassificationReport,
  getConfusionMatrix,
  getROCCurve,
  getTradingResults,
  // Viz — paper figures (Fig. 1–11)
  getVizFig1,
  getVizFig2,
  getVizFig3,
  getVizFig4,
  getVizFig5,
  getVizFig6,
  getVizFig7,
  getVizFig8,
  getVizFig9,
  getVizFig10,
  getVizFig11,
  // Viz — extended figures
  getVizPredVsActual,
  getVizLossCurves,
  getVizConfusionMatrix,
  getVizTradingReturns,
  getVizWalkforward,
  getVizRocCurves,
  getVizFigClassificationTable,
  getVizFigWalkforward,
  getVizFigCumulativeReturn,
  // Polling
  pollJob,
};