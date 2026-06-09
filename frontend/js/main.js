/**
 * frontend/js/main.js
 * ====================
 * Dashboard controller for VNSP — Task 8.3
 *
 * Responsibilities:
 *  - DOMContentLoaded init (navigation, status check, event listeners)
 *  - Section switching
 *  - Experiment matrix building (40 cells)
 *  - Event handlers for all buttons in all sections
 *  - Table rendering helpers
 *  - Table sort (click column header)
 *  - Auto-refresh matrix every 30 s when experiments are running
 *
 * Dependencies (loaded before this file in index.html):
 *  Chart.js 4.4.2  ·  api.js  ·  charts.js
 *
 * Namespace: window.VNSP.main
 */

'use strict';

// Shorthand aliases (set after DOMContentLoaded)
let api, ui, charts;

// =============================================================================
// CONSTANTS
// =============================================================================

const MODELS      = ['DNN', 'RNN', 'GRU', 'LSTM', 'BiLSTM'];
const TICKERS     = ['VCB', 'VIC'];
const CURRENCIES  = ['VND', 'USD'];
const FOLDS       = [1, 2, 3];
const FOLD_LABELS = { 1: 'Fold 1 (2018)', 2: 'Fold 2 (2020)', 3: 'Fold 3 (2022–24)' };

/** 8 column conditions for the matrix (matches api_experiments.py order) */
const MATRIX_CONDITIONS = [
  { ticker:'VCB', currency:'VND', wavelet:true  },
  { ticker:'VCB', currency:'VND', wavelet:false },
  { ticker:'VCB', currency:'USD', wavelet:true  },
  { ticker:'VCB', currency:'USD', wavelet:false },
  { ticker:'VIC', currency:'VND', wavelet:true  },
  { ticker:'VIC', currency:'VND', wavelet:false },
  { ticker:'VIC', currency:'USD', wavelet:true  },
  { ticker:'VIC', currency:'USD', wavelet:false },
];

/** Auto-refresh interval for matrix (ms) */
const MATRIX_REFRESH_MS = 30_000;
let   _matrixTimer = null;


// =============================================================================
// UTILITIES
// =============================================================================

/** Build experiment ID string from parts. */
function buildExpId(ticker, currency, useWavelet, model, task) {
  const cond = (useWavelet === true || useWavelet === 'true') ? 'wavelet' : 'nowave';
  return `${ticker}_${currency}_${cond}_${model}_${task}`;
}

/** Read value from a <select> element by ID. */
const sel = (id) => document.getElementById(id)?.value ?? '';

/** Show/hide an element by toggling .hidden class. */
const setVisible = (id, visible) =>
  document.getElementById(id)?.classList.toggle('hidden', !visible);

/** Insert a base64 image into an img-container div. */
function renderImage(containerId, base64Src, alt = '') {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = base64Src
    ? `<img src="${base64Src}" alt="${alt}" style="max-width:100%;border-radius:4px;" />`
    : `<div class="img-placeholder">No image data.</div>`;
}

/**
 * Generic table body renderer.
 * @param {string}   tbodyId
 * @param {object[]} rows      array of plain objects
 * @param {Array<{key, label, fmt}>} cols   column definitions
 */
function renderTableBody(tbodyId, rows, cols) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  if (!rows || !rows.length) {
    tbody.innerHTML = `<tr><td colspan="${cols.length}" class="table-empty">No data.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(row =>
    `<tr>${cols.map(c => {
      const v = row[c.key] ?? '—';
      const fmt = c.fmt ? c.fmt(v, row) : v;
      return `<td class="${c.cls ?? ''}">${fmt}</td>`;
    }).join('')}</tr>`
  ).join('');
}

/** Format a float to N decimal places (or '—' if null). */
const fmt = (n, d = 4) => (n == null ? '—' : Number(n).toFixed(d));
const fmtPct = (n, d = 2) => (n == null ? '—' : `${(Number(n) * 100).toFixed(d)}%`);
const fmtNum = (n) => (n == null ? '—' : Number(n).toLocaleString('en-US', { maximumFractionDigits: 4 }));


// =============================================================================
// TABLE SORT
// =============================================================================

/** Attach click-to-sort to all .sortable-col <th> elements in a table. */
function attachTableSort(tableId) {
  const table = document.getElementById(tableId);
  if (!table) return;

  table.querySelectorAll('th.sortable-col').forEach(th => {
    th.addEventListener('click', () => {
      const col   = th.dataset.col;
      const tbody = table.querySelector('tbody');
      if (!tbody || !col) return;

      // Determine sort direction
      const asc = !th.classList.contains('sort-asc');
      table.querySelectorAll('th.sortable-col').forEach(h => {
        h.classList.remove('sort-asc', 'sort-desc');
      });
      th.classList.add(asc ? 'sort-asc' : 'sort-desc');

      const colIdx = [...th.parentElement.children].indexOf(th);
      const rows   = [...tbody.querySelectorAll('tr')];

      rows.sort((a, b) => {
        const va = a.children[colIdx]?.textContent.trim() ?? '';
        const vb = b.children[colIdx]?.textContent.trim() ?? '';
        const na = parseFloat(va.replace(/[^0-9.-]/g, ''));
        const nb = parseFloat(vb.replace(/[^0-9.-]/g, ''));
        // Numeric sort if both parse, else string sort
        const cmp = (!isNaN(na) && !isNaN(nb))
          ? na - nb
          : va.localeCompare(vb);
        return asc ? cmp : -cmp;
      });

      rows.forEach(r => tbody.appendChild(r));
    });
  });
}


// =============================================================================
// NAVIGATION
// =============================================================================

/**
 * Switch visible section and update sidebar active state.
 * @param {string} sectionName  e.g. 'data', 'experiments', 'regression'
 */
function switchSection(sectionName) {
  // Deactivate all
  document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(a => a.classList.remove('active'));

  // Activate target
  document.getElementById(`section-${sectionName}`)?.classList.add('active');
  document.querySelector(`.nav-item[data-section="${sectionName}"]`)?.classList.add('active');
}


// =============================================================================
// API STATUS CHECK
// =============================================================================

async function checkApiStatus() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (!dot || !text) return;

  dot.className  = 'status-dot checking';
  text.textContent = 'Checking…';

  try {
    const res = await fetch(`${location.origin}/health`);
    if (res.ok) {
      dot.className    = 'status-dot online';
      text.textContent = 'API Online';
    } else {
      throw new Error(`${res.status}`);
    }
  } catch {
    dot.className    = 'status-dot offline';
    text.textContent = 'API Offline';
  }
}


// =============================================================================
// SECTION: DATA MANAGEMENT
// =============================================================================

/** Render file status cards from /api/data/status */
async function loadDataStatus() {
  const grid = document.getElementById('data-status-grid');
  if (!grid) return;

  try {
    const data  = await api.checkDataStatus();
    const { raw, processed } = data;

    const rawItems = Object.entries(raw).map(([k, ok]) =>
      `<div class="status-item ${ok ? 'ok' : 'missing'}">
         <span class="file-icon"></span>
         <span>${k}${k !== 'USDVND' ? '_raw.csv' : '.csv'}</span>
       </div>`
    );

    const pklItems = Object.entries(processed).map(([k, ok]) =>
      `<div class="status-item ${ok ? 'ok' : 'missing'}">
         <span class="file-icon"></span>
         <span>${k}.pkl</span>
       </div>`
    );

    grid.innerHTML = [...rawItems, ...pklItems].join('');
  } catch (e) {
    grid.innerHTML = `<div class="loading-placeholder">Failed to load status.</div>`;
  }
}

async function loadRawData() {
  const ticker   = sel('data-ticker');
  const currency = sel('data-currency');
  ui.showSpinner(`Loading ${ticker} raw data…`);

  try {
    const data  = await api.getRawData(ticker, currency, { limit: 200 });
    const cols  = ['Date','Open','High','Low','Close','Volume'];
    const apiKeys = ['Date','Open','High','Low','Close','Volume'];

    document.getElementById('raw-data-count').textContent = `${data.n_rows} rows`;

    renderTableBody('raw-data-tbody', data.data,
      apiKeys.map((k, i) => ({
        key: k.toLowerCase(),
        label: cols[i],
        fmt: (v) => {
          if (k === 'Date') return v;
          if (k === 'Volume') return Number(v).toLocaleString();
          return currency === 'USD' ? fmt(v, 4) : Number(v).toLocaleString('vi-VN');
        },
      }))
    );
  } catch (_) {
    /* toast shown by api.js */
  } finally {
    ui.hideSpinner();
  }
}

async function loadFeatures() {
  const ticker     = sel('data-ticker');
  const currency   = sel('data-currency');
  const useWavelet = sel('data-wavelet') === 'true';
  ui.showSpinner('Loading features…');

  try {
    const data = await api.getFeatures(ticker, currency, useWavelet, 50);
    document.getElementById('features-info').textContent =
      `${data.n_rows} rows · ${data.feature_names.length} features · ${data.date_range?.start} → ${data.date_range?.end}`;

    const featureCols = data.feature_names;

    // Build header
    const thead = document.getElementById('features-thead');
    thead.innerHTML = `<tr>${['Date', ...featureCols].map(c => `<th>${c}</th>`).join('')}</tr>`;

    // Build rows from sample
    const tbody = document.getElementById('features-tbody');
    if (!data.sample?.length) {
      tbody.innerHTML = `<tr><td colspan="${featureCols.length + 1}" class="table-empty">No sample data.</td></tr>`;
      return;
    }
    tbody.innerHTML = data.sample.map(row => {
      const cols = [row.time ?? row.Date ?? '—', ...featureCols.map(k => fmt(row[k], 4))];
      return `<tr>${cols.map(v => `<td>${v}</td>`).join('')}</tr>`;
    }).join('');

  } catch (_) { /* toast shown */ } finally { ui.hideSpinner(); }
}

async function loadDeviationPlot() {
  const ticker   = sel('data-ticker');
  const currency = sel('data-currency');
  ui.showSpinner('Generating deviation plot…');
  try {
    const data = await api.getDeviationPlot(ticker, currency);
    renderImage('deviation-plot-container', data.image, `${ticker} Deviation Plot`);
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}

async function triggerPreprocess() {
  const ticker     = sel('data-ticker');
  const currency   = sel('data-currency');
  const useWavelet = sel('data-wavelet') === 'true';
  ui.showSpinner(`Preprocessing ${ticker} ${currency}…`);
  try {
    const res = await api.triggerPreprocess(ticker, currency, useWavelet);
    ui.showToast('success', 'Preprocessing done',
      `${res.n_features} features · ${res.n_rows} rows`);
    await loadDataStatus();
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}


// =============================================================================
// SECTION: EXPERIMENTS
// =============================================================================

/** Build the 40-cell experiment matrix. */
async function loadExperimentMatrix() {
  const taskFilter = sel('matrix-task-filter') || 'regression';
  const body = document.getElementById('matrix-body');
  if (!body) return;
  body.innerHTML = `<div class="loading-placeholder">Loading matrix…</div>`;

  try {
    const data        = await api.getExperimentMatrix({ task: taskFilter });
    const experiments = data.experiments ?? [];

    // Update summary pills
    const done = experiments.filter(e => e.status === 'done').length;
    const part = experiments.filter(e => e.status === 'partial').length;
    const pend = experiments.filter(e => e.status === 'pending').length;
    document.getElementById('m-total').textContent   = data.total ?? experiments.length;
    document.getElementById('m-done').textContent    = done;
    document.getElementById('m-partial').textContent = part;
    document.getElementById('m-pending').textContent = pend;
    document.getElementById('m-error').textContent   =
      experiments.filter(e => e.status === 'error').length;

    // Build lookup: exp_id → experiment
    const lookup = Object.fromEntries(experiments.map(e => [e.exp_id, e]));

    // Build HTML rows
    let html = '';
    for (const model of MODELS) {
      html += `<div class="matrix-row"><div class="matrix-row-label">${model}</div>`;
      for (const cond of MATRIX_CONDITIONS) {
        const condStr = cond.wavelet ? 'wavelet' : 'nowave';
        const expId   = buildExpId(cond.ticker, cond.currency, cond.wavelet, model, taskFilter);
        const exp     = lookup[expId];
        const status  = exp?.status  ?? 'pending';
        const folds   = exp ? `${exp.folds_done}/${exp.folds_total}` : '0/3';
        const title   = `${cond.ticker} / ${cond.currency} / ${condStr} / ${model} / ${taskFilter}`;
        html += `<div class="matrix-cell ${status}"
            id="mcell-${expId}"
            data-exp="${expId}"
            title="${title}"
            onclick="VNSP.main.handleCellClick('${expId}', '${taskFilter}')">
          <span class="cell-folds">${folds}</span>
        </div>`;
      }
      html += '</div>';
    }
    body.innerHTML = html;

    // Schedule auto-refresh if any experiments are still running/partial
    const hasRunning = experiments.some(e => e.status === 'partial' || e.status === 'pending');
    _scheduleMatrixRefresh(hasRunning ? MATRIX_REFRESH_MS : 0);

  } catch (_) {
    body.innerHTML = `<div class="loading-placeholder">Failed to load matrix.</div>`;
  }
}

/** Click on a matrix cell → navigate to results section with that exp pre-selected. */
function handleCellClick(expId, task = 'regression') {
  const parts = expId.split('_');
  if (parts.length < 5) return;
  const [ticker, currency] = parts;

  if (task === 'regression') {
    document.getElementById('reg-ticker').value   = ticker;
    document.getElementById('reg-currency').value = currency;
    switchSection('regression');
    loadRegression();
  } else {
    document.getElementById('cls-ticker').value   = ticker;
    document.getElementById('cls-currency').value = currency;
    switchSection('classification');
  }
}

/** Auto-refresh matrix: enable periodic poll if delay > 0, else cancel. */
function _scheduleMatrixRefresh(delayMs) {
  if (_matrixTimer) clearInterval(_matrixTimer);
  _matrixTimer = null;
  if (delayMs > 0) {
    _matrixTimer = setInterval(loadExperimentMatrix, delayMs);
  }
}

/** Show/update job progress area (run experiment or HPO). */
function _showJobArea(areaId, progressId, jobIdElId, jobId, progress) {
  setVisible(areaId, true);
  document.getElementById(progressId).textContent = progress;
  document.getElementById(jobIdElId).textContent  = jobId;
}

async function submitRunExperiment() {
  const config = {
    ticker      : sel('run-ticker'),
    currency    : sel('run-currency'),
    use_wavelet : sel('run-wavelet') === 'true',
    model       : sel('run-model'),
    task        : sel('run-task'),
    fold        : sel('run-fold') ? parseInt(sel('run-fold'), 10) : null,
  };
  if (!config.fold) delete config.fold;

  try {
    const res = await api.runExperiment(config);
    ui.showToast('info', 'Experiment started', `Job ID: ${res.job_id}`);
    _showJobArea('run-job-area', 'run-job-progress', 'run-job-id', res.job_id, '0/3 folds starting…');

    // Show in header badge
    setVisible('job-badge', true);
    document.getElementById('job-badge-text').textContent = `Experiment ${res.exp_id}…`;

    api.pollJob(
      res.job_id,
      (s) => _showJobArea('run-job-area', 'run-job-progress', 'run-job-id', res.job_id, s.progress),
      (s) => {
        _showJobArea('run-job-area', 'run-job-progress', 'run-job-id', res.job_id, '✓ Done');
        ui.showToast('success', 'Experiment complete', s.exp_id);
        setVisible('job-badge', false);
        loadExperimentMatrix();   // refresh matrix
      },
      (err) => {
        _showJobArea('run-job-area', 'run-job-progress', 'run-job-id', res.job_id, `✕ Error: ${err}`);
        ui.showToast('error', 'Experiment failed', err);
        setVisible('job-badge', false);
      }
    );
  } catch (_) { /* toast shown by api.js */ }
}

async function submitRunHPO() {
  const config = {
    ticker      : sel('hpo-ticker'),
    currency    : sel('hpo-currency'),
    use_wavelet : sel('hpo-wavelet') === 'true',
    n_trials    : parseInt(sel('hpo-trials'), 10) || 30,
  };

  try {
    const res = await api.runHPO(config);
    ui.showToast('info', 'HPO started', `${res.label} · ${config.n_trials} trials/fold`);
    _showJobArea('hpo-job-area', 'hpo-job-progress', 'hpo-job-id', res.job_id, 'HPO running (3 folds)…');

    setVisible('job-badge', true);
    document.getElementById('job-badge-text').textContent = `HPO ${res.label}…`;

    api.pollJob(
      res.job_id,
      (s) => _showJobArea('hpo-job-area', 'hpo-job-progress', 'hpo-job-id', res.job_id, s.progress),
      () => {
        _showJobArea('hpo-job-area', 'hpo-job-progress', 'hpo-job-id', res.job_id, '✓ HPO Done');
        ui.showToast('success', 'HPO complete', `best_params.json saved`);
        setVisible('job-badge', false);
      },
      (err) => {
        _showJobArea('hpo-job-area', 'hpo-job-progress', 'hpo-job-id', res.job_id, `✕ ${err}`);
        ui.showToast('error', 'HPO failed', err);
        setVisible('job-badge', false);
      }
    );
  } catch (_) { /* toast */ }
}


// =============================================================================
// SECTION: REGRESSION
// =============================================================================

async function loadRegression() {
  const ticker   = sel('reg-ticker');
  const currency = sel('reg-currency');
  ui.showSpinner('Loading regression results…');

  try {
    const data = await api.getComparisonTable(ticker, currency, 'regression');

    const models       = data.models ?? [];
    const beforeWave   = data.before_wavelet ?? {};
    const afterWave    = data.after_wavelet  ?? {};

    const tbody = document.getElementById('comparison-tbody');
    tbody.innerHTML = models.map((m, i) => `
      <tr>
        <td style="font-family:var(--font-mono);font-weight:600">${m}</td>
        <td>${fmtNum(beforeWave.MSE?.[i])}</td>
        <td>${fmtNum(beforeWave.MAE?.[i])}</td>
        <td>${fmt(beforeWave.MAPE?.[i], 2)}</td>
        <td class="text-success">${fmtNum(afterWave.MSE?.[i])}</td>
        <td class="text-success">${fmtNum(afterWave.MAE?.[i])}</td>
        <td class="text-success">${fmt(afterWave.MAPE?.[i], 2)}</td>
      </tr>`
    ).join('');

    // Render MSE comparison chart (Fig. 11 equivalent)
    charts.renderMSEComparison('chart-pred-actual', {   // reuse canvas for now
      models, before_wavelet: beforeWave, after_wavelet: afterWave, metric: 'MSE',
      placeholderId: 'pred-chart-placeholder',
    });

    ui.showToast('success', 'Regression results loaded', `${models.length} models`);
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}

async function loadPredChart() {
  const ticker     = sel('reg-ticker');
  const currency   = sel('reg-currency');
  const model      = sel('pred-model');
  const useWavelet = sel('pred-wavelet') === 'true';
  const fold       = parseInt(sel('reg-fold'), 10) || 3;
  const expId      = buildExpId(ticker, currency, useWavelet, model, 'regression');
  ui.showSpinner('Loading predictions…');

  try {
    const data = await api.getPredictions(expId, fold);
    charts.renderPredictedVsActual('chart-pred-actual', {
      ...data,
      currency,
      ticker,
      placeholderId: 'pred-chart-placeholder',
    });
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}

async function loadLossCurves() {
  const ticker     = sel('reg-ticker');
  const currency   = sel('reg-currency');
  const model      = sel('pred-model');
  const useWavelet = sel('pred-wavelet') === 'true';
  const fold       = parseInt(sel('reg-fold'), 10) || 1;
  const expId      = buildExpId(ticker, currency, useWavelet, model, 'regression');
  ui.showSpinner('Loading loss curves…');

  try {
    const data = await api.getLossCurves(expId, fold);
    charts.renderLossCurves('chart-loss-curves', { ...data, placeholderId: 'loss-placeholder' });
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}


// =============================================================================
// SECTION: CLASSIFICATION
// =============================================================================

async function loadClassification() {
  const ticker     = sel('cls-ticker');
  const currency   = sel('cls-currency');
  const useWavelet = sel('cls-wavelet') === 'true';
  const model      = sel('cls-model');
  const fold       = parseInt(sel('cls-fold'), 10) || 3;
  const expId      = buildExpId(ticker, currency, useWavelet, model, 'classification');
  ui.showSpinner('Loading classification results…');

  try {
    // Load all three classification views in parallel
    const [reportData, cmData, rocData] = await Promise.allSettled([
      api.getClassificationReport(expId),
      api.getConfusionMatrix(expId, fold),
      api.getROCCurve(expId, fold),
    ]);

    // Classification report table + metric cards
    if (reportData.status === 'fulfilled') {
      const folds = reportData.value.folds ?? [];
      const agg   = reportData.value.aggregated ?? {};

      renderTableBody('cls-report-tbody', folds,
        ['fold_idx', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC'].map(k => ({
          key: k === 'fold_idx' ? 'fold_idx' : 'metrics',
          label: k,
          fmt: k === 'fold_idx'
            ? (v, row) => FOLD_LABELS[row.fold_idx] ?? row.fold_idx
            : (_, row) => row.status === 'done' ? fmt(row.metrics?.[k], 4) : '—',
        }))
      );

      // Update metric cards (use aggregated means)
      const metricIds = { Accuracy: 'mc-accuracy', Precision: 'mc-precision',
                          Recall: 'mc-recall',    F1: 'mc-f1', AUC_ROC: 'mc-auc' };
      for (const [metric, cardId] of Object.entries(metricIds)) {
        const el = document.getElementById(cardId);
        if (!el) continue;
        const val = agg[`${metric}_mean`];
        el.querySelector('.metric-value').textContent = val != null ? fmt(val, 4) : '—';
      }
    }

    // Confusion matrix (HTML render)
    if (cmData.status === 'fulfilled') {
      charts.renderConfusionMatrix('cm-container', cmData.value);
    }

    // ROC curve (Chart.js)
    if (rocData.status === 'fulfilled') {
      charts.renderROCCurve('chart-roc', {
        ...rocData.value,
        label   : `${model} ${useWavelet ? '(Wave)' : '(No-Wave)'}`,
        placeholderId: 'roc-placeholder',
      });
    }

    ui.showToast('success', 'Classification results loaded');
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}


// =============================================================================
// SECTION: TRADING
// =============================================================================

async function loadTrading() {
  const ticker   = sel('trade-ticker');
  const currency = sel('trade-currency');
  const fold     = parseInt(sel('trade-fold'), 10) || 3;
  ui.showSpinner('Loading trading simulation…');

  try {
    const data = await api.getTradingResults(ticker, currency, fold);
    const rows = data.data ?? [];

    // Render summary bar chart
    if (rows.length) {
      charts.renderCumulativeReturn('chart-trading-returns', {
        rows,
        placeholderId: 'trading-placeholder',
      });
    }

    // Render table
    renderTableBody('trading-tbody', rows, [
      { key: 'model_name',          label: 'Model',         fmt: v => `<strong>${v}</strong>` },
      { key: 'use_wavelet',         label: 'Wavelet',       fmt: v => v ? '✓ Wave' : 'No-Wave' },
      { key: 'Cumulative_Return',   label: 'Cum. Return',   fmt: v => fmtPct(v) },
      { key: 'BuyHold_Return',      label: 'Buy & Hold',    fmt: v => fmtPct(v) },
      { key: 'Sharpe_Ratio',        label: 'Sharpe',        fmt: v => fmt(v, 3) },
      { key: 'Max_Drawdown',        label: 'Max DD',        fmt: v => fmtPct(v) },
      { key: 'Win_Rate',            label: 'Win Rate',      fmt: v => fmtPct(v) },
    ]);

    attachTableSort('trading-table');
    ui.showToast('success', 'Trading results loaded', `${rows.length} model variants`);
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}


// =============================================================================
// SECTION: FIGURES
// =============================================================================

async function loadFig11() {
  const ticker   = sel('fig11-ticker');
  const currency = sel('fig11-currency');
  ui.showSpinner('Generating Fig. 11…');
  try {
    const res = await api.getVizFig11(ticker, currency);
    renderImage('fig11-container', res.image, 'Fig. 11 MSE comparison');
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}

async function loadVndVsUsd() {
  ui.showSpinner('Loading VND vs USD comparison…');
  try {
    const data = await api.getVndVsUsd();
    const { columns, data: rows } = data;

    // Build dynamic headers and rows
    const thead = document.getElementById('vnd-usd-thead');
    const tbody = document.getElementById('vnd-usd-tbody');
    if (!columns?.length) {
      tbody.innerHTML = `<tr><td class="table-empty">No data.</td></tr>`;
      return;
    }
    thead.innerHTML = `<tr>${columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
    tbody.innerHTML = rows.map(row =>
      `<tr>${columns.map(c => `<td>${row[c] != null ? fmt(row[c], 4) : '—'}</td>`).join('')}</tr>`
    ).join('');
  } catch (_) { /* toast */ } finally { ui.hideSpinner(); }
}


// =============================================================================
// EVENT LISTENERS — attach all button clicks
// =============================================================================

function attachEventListeners() {
  const on = (id, fn) => document.getElementById(id)?.addEventListener('click', fn);

  // Navigation
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      switchSection(item.dataset.section);
    });
  });

  // Data section
  on('btn-refresh-status',  loadDataStatus);
  on('btn-load-raw',        loadRawData);
  on('btn-load-features',   loadFeatures);
  on('btn-load-deviation',  loadDeviationPlot);
  on('btn-preprocess',      triggerPreprocess);

  // Experiments section
  on('btn-refresh-matrix',  loadExperimentMatrix);
  on('btn-run-experiment',  submitRunExperiment);
  on('btn-run-hpo',         submitRunHPO);
  document.getElementById('matrix-task-filter')
    ?.addEventListener('change', loadExperimentMatrix);

  // Regression section
  on('btn-load-regression', loadRegression);
  on('btn-load-pred-chart', loadPredChart);
  on('btn-load-loss',       loadLossCurves);

  // Classification section
  on('btn-load-classification', loadClassification);

  // Trading section
  on('btn-load-trading', loadTrading);

  // Figures section
  on('btn-load-fig11',   loadFig11);
  on('btn-load-vnd-usd', loadVndVsUsd);

  // Enable sort on static tables
  attachTableSort('raw-data-table');
  attachTableSort('trading-table');
}


// =============================================================================
// INIT — DOMContentLoaded
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Bind namespace shortcuts after all scripts are loaded
  api    = window.VNSP?.api;
  ui     = window.VNSP?.ui;
  charts = window.VNSP?.charts;

  if (!api || !ui || !charts) {
    console.error('[main.js] VNSP.api / ui / charts not loaded. Check script order.');
    return;
  }

  attachEventListeners();

  // Initial startup: check API status + data file status
  checkApiStatus();
  loadDataStatus();
});


// =============================================================================
// NAMESPACE EXPORT
// =============================================================================

window.VNSP       = window.VNSP || {};
window.VNSP.main  = {
  switchSection,
  handleCellClick,
  loadExperimentMatrix,
  // Expose helpers for inline onclick (matrix cells)
  buildExpId,
};