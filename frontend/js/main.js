/**
 * frontend/js/main.js
 * ====================
 * Dashboard controller for VNSP — Redesign 2026-06
 *
 * Sections: Data | Pipeline | Regression | Classification | Trading
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

    // Raw table
    if (rawRes.status === 'fulfilled' && rawRes.value?.data) {
      document.getElementById('raw-count').textContent = `${rawRes.value.n_rows} rows`;
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

// =============================================================================
// SECTION 2 — PIPELINE ANALYSIS
// =============================================================================

async function loadArchitecture() {
  const ticker  = sel('pipe-ticker');
  const model   = sel('pipe-model');
  const task    = sel('pipe-task');
  const wavelet = sel('pipe-wavelet') === 'true';
  const fold    = parseInt(sel('pipe-fold'), 10) || 1;
  const cond    = wavelet ? 'wavelet' : 'nowave';
  const expId   = buildExpId(ticker, 'VND', wavelet, model, task);

  setHTML('arch-title', `Architecture — ${model} · ${task} · ${wavelet ? 'With Wavelet' : 'No Wavelet'} · ${FOLD_LABELS[fold]}`);
  setHTML('arch-body', '<div class="loading-placeholder">Loading hyperparameters…</div>');
  setHTML('flow-body', '<div class="loading-placeholder">Loading…</div>');

  ui.showSpinner('Loading architecture…');
  try {
    // Load actual best_params from API
    const paramsRes = await api.getBestParams(expId, fold).catch(() => null);
    const params    = paramsRes?.best_params ?? {};

    // Feature count: wavelet ~8, nowave = 5 (from preprocessing)
    // Load actual feature count from features API
    const featRes = await api.getFeatures(ticker, 'VND', wavelet, 1).catch(() => null);
    const nFeatures = featRes?.feature_names
      ? featRes.feature_names.filter(f => f !== 'Close').length
      : (wavelet ? 8 : 5);

    // Render architecture diagram
    setHTML('arch-body', _renderArchDiagram(model, task, wavelet, params, nFeatures));
    // Render data flow
    setHTML('flow-body', _renderDataFlow(model, task, wavelet, params, nFeatures));

  } catch (e) {
    setHTML('arch-body', `<div class="loading-placeholder text-red">Error: ${e.message}</div>`);
  } finally { ui.hideSpinner(); }
}

/** Render architecture visualization as HTML */
function _renderArchDiagram(model, task, wavelet, params, nFeatures) {
  const seqLen  = params.sequence_length ?? 20;
  const hidden  = params.hidden_units    ?? 64;
  const nLayers = params.num_layers      ?? 1;
  const dropout = params.dropout_rate    ?? 0.2;
  const outDim  = task === 'regression' ? 1 : 1;
  const outAct  = task === 'regression' ? 'Linear' : 'Sigmoid';

  // Build model-specific layers HTML
  let modelLayersHtml = '';

  if (model === 'DNN') {
    modelLayersHtml = `
      <div class="arch-block model">
        <div class="arch-label">Flatten</div>
        <div class="arch-name">Dense Head</div>
        <div class="arch-dim">${seqLen}×${nFeatures} → ${hidden}</div>
      </div>
      <span class="arch-arrow">→</span>
      <div class="arch-block model">
        <div class="arch-label">${nLayers} Dense Layer(s)</div>
        <div class="arch-name">FC + ReLU + BN</div>
        <div class="arch-dim">hidden=${hidden}</div>
      </div>`;
  } else if (model === 'RNN') {
    modelLayersHtml = `
      <div class="arch-block model">
        <div class="arch-label">RNN × ${nLayers}</div>
        <div class="arch-name">RNN (relu)</div>
        <div class="arch-dim">in=${nFeatures} h=${hidden}</div>
      </div>
      <span class="arch-arrow">→</span>
      <div class="arch-block model">
        <div class="arch-label">Last timestep</div>
        <div class="arch-name">FC Layer</div>
        <div class="arch-dim">${hidden} → ${hidden}</div>
      </div>`;
  } else if (model === 'GRU') {
    modelLayersHtml = `
      <div class="arch-block model">
        <div class="arch-label">GRU × ${nLayers}</div>
        <div class="arch-name">GRU + BN1d</div>
        <div class="arch-dim">in=${nFeatures} h=${hidden}</div>
      </div>
      <span class="arch-arrow">→</span>
      <div class="arch-block model">
        <div class="arch-label">Dropout(${dropout})</div>
        <div class="arch-name">FC + ReLU</div>
        <div class="arch-dim">${hidden} → ${hidden/2|0}</div>
      </div>`;
  } else if (model === 'LSTM') {
    modelLayersHtml = `
      <div class="arch-block model">
        <div class="arch-label">LSTM × ${nLayers}</div>
        <div class="arch-name">LSTM + BN1d</div>
        <div class="arch-dim">in=${nFeatures} h=${hidden}</div>
      </div>
      <span class="arch-arrow">→</span>
      <div class="arch-block model">
        <div class="arch-label">3 Dense Layers</div>
        <div class="arch-name">FC256→FC64→FC1</div>
        <div class="arch-dim">dropout=${dropout}</div>
      </div>`;
  } else if (model === 'BiLSTM') {
    if (wavelet) {
      // Dual branch for wavelet case
      const halfFeat = Math.ceil(nFeatures / 2);
      const fuseSize = hidden * 4;  // 2 branches × 2 directions
      modelLayersHtml = `
        <div class="arch-dual-branch">
          <div class="arch-branch-col">
            <div class="arch-branch-label">A1 Branch (${halfFeat} feats)</div>
            <div class="arch-block branch">
              <div class="arch-label">BiLSTM × ${nLayers}</div>
              <div class="arch-name">Approx</div>
              <div class="arch-dim">→ ${hidden*2}</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;padding-top:20px;color:#3a3a5a">⊕</div>
          <div class="arch-branch-col">
            <div class="arch-branch-label">D1 Branch (${nFeatures-halfFeat} feats)</div>
            <div class="arch-block branch">
              <div class="arch-label">BiLSTM × ${nLayers}</div>
              <div class="arch-name">Detail</div>
              <div class="arch-dim">→ ${hidden*2}</div>
            </div>
          </div>
        </div>
        <span class="arch-arrow">→</span>
        <div class="arch-block model">
          <div class="arch-label">Concatenate</div>
          <div class="arch-name">Fused</div>
          <div class="arch-dim">${fuseSize}</div>
        </div>`;
    } else {
      modelLayersHtml = `
        <div class="arch-block model">
          <div class="arch-label">BiLSTM × ${nLayers}</div>
          <div class="arch-name">Bidirectional</div>
          <div class="arch-dim">in=${nFeatures} h=${hidden}×2</div>
        </div>
        <span class="arch-arrow">→</span>
        <div class="arch-block model">
          <div class="arch-label">FC Layers</div>
          <div class="arch-name">Dense Head</div>
          <div class="arch-dim">${hidden*2} → ${hidden}</div>
        </div>`;
    }
  }

  // Params chips
  const paramChips = Object.entries(params).filter(([k]) => k !== 'n_features').map(([k, v]) =>
    `<div class="param-chip">
      <div class="param-key">${k.replace(/_/g,' ')}</div>
      <div class="param-val">${v}</div>
    </div>`
  ).join('');

  return `
    <div class="arch-flow">
      <div class="arch-block input">
        <div class="arch-label">Input Sequence</div>
        <div class="arch-name">Sequences</div>
        <div class="arch-dim">${seqLen} × ${nFeatures}</div>
      </div>
      <span class="arch-arrow">→</span>
      ${modelLayersHtml}
      <span class="arch-arrow">→</span>
      <div class="arch-block output">
        <div class="arch-label">Output Head</div>
        <div class="arch-name">${outAct}</div>
        <div class="arch-dim">→ ${outDim}</div>
      </div>
    </div>
    ${paramChips ? `<div class="section-label mt-12">Best Hyperparameters</div><div class="params-grid">${paramChips}</div>` : ''}`;
}

/** Render data processing flow description */
function _renderDataFlow(model, task, wavelet, params, nFeatures) {
  const seqLen  = params.sequence_length ?? 20;
  const steps = wavelet ? [
    { icon: '1', label: 'Raw OHLCV (5)', desc: 'Open, High, Low, Close, Volume · VND' },
    { icon: '2', label: '+ Deviation', desc: 'Deviation = Close − Open (buy/sell pressure)' },
    { icon: '3', label: 'SWT db4 Level-1', desc: 'Each feature → Approx (A1) + Detail (D1) = 10 coefficients' },
    { icon: '4', label: 'Feature Selection', desc: `Pearson |r| > 0.95 threshold · fit on Fold 1 train (≤2017) · ${nFeatures} features kept` },
    { icon: '5', label: 'StandardScaler / RobustScaler', desc: 'Price/Deviation → StandardScaler · Volume → RobustScaler · Open_Approx unscaled' },
    { icon: '6', label: `Sliding Window (L=${seqLen})`, desc: `Each sample: ${seqLen} × ${nFeatures} array → predict ${task === 'regression' ? 'Close(t+1)' : 'direction of next week'}` },
  ] : [
    { icon: '1', label: 'Raw OHLCV (5)', desc: 'Open, High, Low, Close, Volume · VND' },
    { icon: '2', label: '+ Deviation', desc: 'Deviation = Close − Open (buy/sell pressure)' },
    { icon: '3', label: '5 Raw Features', desc: 'Open, High, Low, Volume, Deviation · Close as target' },
    { icon: '4', label: 'StandardScaler / RobustScaler', desc: 'Price/Deviation → StandardScaler · Volume → RobustScaler · Open unscaled' },
    { icon: '5', label: `Sliding Window (L=${seqLen})`, desc: `Each sample: ${seqLen} × ${nFeatures} array → predict ${task === 'regression' ? 'Close(t+1)' : 'direction of next week'}` },
  ];

  const taskDesc = task === 'regression'
    ? 'Loss: MSE · Metrics: MSE, MAE, MAPE, RMSE, R² · Output: price value (inverse-transformed to VND)'
    : 'Loss: BCELoss + Sigmoid · Metrics: Accuracy, F1, AUC-ROC · Output: P(UP) probability · Target: weekly direction T2→T6';

  return `
    <div class="arch-flow" style="flex-direction:column;align-items:flex-start;gap:8px">
      ${steps.map(s => `
        <div style="display:flex;align-items:center;gap:12px;width:100%">
          <div style="width:22px;height:22px;border-radius:50%;background:var(--blue-dim);
            color:var(--blue);font-size:11px;font-weight:700;display:flex;align-items:center;
            justify-content:center;flex-shrink:0">${s.icon}</div>
          <div>
            <div style="font-weight:600;font-size:12px">${s.label}</div>
            <div style="font-size:11px;color:var(--fg-dim)">${s.desc}</div>
          </div>
        </div>
        ${s !== steps[steps.length-1] ? '<div style="margin-left:10px;color:var(--border2)">↓</div>' : ''}
      `).join('')}
    </div>
    <div class="mt-12" style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:10px 14px">
      <div class="section-label" style="margin-bottom:6px">Task: ${task.toUpperCase()}</div>
      <div style="font-size:12px;color:var(--fg-dim)">${taskDesc}</div>
    </div>`;
}

// Load static pipeline diagrams when entering pipeline section
async function loadPipelineOverview() {
  _loadVizImage('pipe-fig1', () => api.getVizFig1(),  'Pipeline Framework');
  _loadVizImage('pipe-fig9', () => api.getVizFig9(),  'SWT Decomposition');
}

// =============================================================================
// SECTION 3 — REGRESSION
// =============================================================================

async function loadRegression() {
  const ticker  = sel('reg-ticker');
  const wavelet = sel('reg-wavelet') === 'true';
  const fold    = parseInt(sel('reg-fold'), 10) || 3;
  const wLabel  = wavelet ? 'With Wavelet' : 'No Wavelet';
  document.getElementById('reg-table-label').textContent = `${ticker} · VND · ${wLabel} · ${FOLD_LABELS[fold]}`;

  ui.showSpinner('Loading regression results…');
  try {
    // ── Comparison table (mean across folds from API) ─────────────────────
    const tbl = await api.getComparisonTable(ticker, 'VND', 'regression').catch(() => null);
    if (tbl) _renderRegressionTable(tbl, wavelet);

    // ── Predicted vs Actual — collect all 5 models ────────────────────────
    // dates có thể null với regression (chỉ classification weekly có dates)
    // → dùng index làm X labels nếu dates không có
    const predPromises = MODELS.map(model => {
      const expId = buildExpId(ticker, 'VND', wavelet, model, 'regression');
      return api.getPredictions(expId, fold)
        .then(r => ({
          model,
          wavelet,
          y_true: Array.isArray(r.y_true) ? r.y_true : [],
          y_pred: Array.isArray(r.y_pred) ? r.y_pred : [],
          dates : Array.isArray(r.dates) && r.dates.length ? r.dates : null,
        }))
        .catch(() => null);
    });
    const predResults = (await Promise.allSettled(predPromises))
      .filter(r => r.status === 'fulfilled' && r.value && r.value.y_true.length)
      .map(r => r.value);

    if (predResults.length > 0) {
      charts.buildPredChart('pred-chart', predResults);
    }

    // ── Loss curves — one per model ────────────────────────────────────────
    MODELS.forEach(async model => {
      const expId = buildExpId(ticker, 'VND', wavelet, model, 'regression');
      try {
        const lc = await api.getLossCurves(expId, fold);
        charts.buildLossChart(`loss-chart-${LOSS_CANVAS[model]}`, {
          train_losses: lc.train_losses,
          val_losses  : lc.val_losses,
          best_epoch  : lc.best_epoch,
          model_label : `${model}`,
        });
      } catch { /* skip if not available */ }
    });

  } catch (e) { /* toast */ }
  finally { ui.hideSpinner(); }
}

function _renderRegressionTable(tbl, showWavelet) {
  // tbl from comparison-table API: {models, before_wavelet: {MSE,MAE,...}, after_wavelet: {...}}
  const tbody = document.getElementById('reg-metrics-tbody');
  if (!tbody) return;

  const key   = showWavelet ? 'after_wavelet' : 'before_wavelet';
  const data  = tbl[key];
  const models = tbl.models || MODELS;

  if (!data) {
    tbody.innerHTML = `<tr><td colspan="6" class="table-empty">No data for this condition.</td></tr>`;
    return;
  }

  // Find best values per metric
  const metrics = ['MSE', 'MAE', 'MAPE', 'RMSE', 'R2'];
  const best = {};
  metrics.forEach(m => {
    if (!data[m]) return;
    const vals = data[m].filter(v => v != null);
    if (m === 'R2') best[m] = Math.max(...vals);
    else best[m] = Math.min(...vals);
  });

  tbody.innerHTML = models.map((model, i) => {
    const row = metrics.map(m => {
      const v = data[m]?.[i];
      if (v == null) return '<td>—</td>';
      const isBest = Math.abs(v - best[m]) < 1e-10;
      const cls = isBest ? 'best-val' : '';
      return `<td class="${cls}">${fmt(v)}</td>`;
    }).join('');
    return `<tr><td class="model-name">${model}</td>${row}</tr>`;
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

    // Chart.js line chart
    if (tsRes.status === 'fulfilled' && tsRes.value?.data?.length) {
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
      // Auto-load static content when entering pipeline section
      if (item.dataset.section === 'pipeline') loadPipelineOverview();
    });
  });

  // Section 1: Data
  on('btn-load-data', loadData);

  // Section 2: Pipeline
  on('btn-load-arch', loadArchitecture);

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
window.VNSP.main = { switchSection, loadData, loadArchitecture, loadRegression, loadClassification, loadTrading };