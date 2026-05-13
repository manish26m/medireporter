/**
 * MediReporter v2.0 — Frontend Application
 * Handles: file upload, API calls, tab switching,
 * mode toggle (Standard / High-Accuracy), pipeline animation,
 * risk/confidence rendering, PDF export.
 */
document.addEventListener('DOMContentLoaded', () => {

  // ── Element refs ─────────────────────────────────────────
  const dropZone      = document.getElementById('drop-zone');
  const fileInput     = document.getElementById('file-input');
  const textarea      = document.getElementById('report-text');
  const analyzeBtn    = document.getElementById('analyze-btn');
  const analyzeBtnTx  = document.getElementById('analyze-btn-text');
  const charCount     = document.getElementById('char-count');
  const uiError       = document.getElementById('ui-error');

  const emptyState    = document.getElementById('empty-state');
  const loadingState  = document.getElementById('loading-state');
  const resultsContent= document.getElementById('results-content');

  const dlBtn         = document.getElementById('download-slip-btn');
  const slipContainer = document.getElementById('medical-slip-container');
  const modeToggleBtn = document.getElementById('mode-toggle-btn');
  const modeBanner    = document.getElementById('mode-banner');
  const modeBannerContent = document.getElementById('mode-banner-content');
  const modeLabel     = document.getElementById('mode-label');

  let currentFile  = null;
  let accurateMode = false;   // false = Standard (LSTM+BART+BioBERT), true = High-Accuracy (BART+BioBERT)

  // ── Char counter ──────────────────────────────────────────
  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length.toLocaleString() + ' characters';
  });

  // ── File Upload ───────────────────────────────────────────
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

  function handleFile(file) {
    hideError();
    const ok = file.type === 'text/plain' || file.type === 'application/pdf'
      || file.name.endsWith('.txt') || file.name.endsWith('.pdf');
    if (!ok) { showError('Please upload a .txt or .pdf file.'); return; }
    currentFile = file;
    document.getElementById('upload-text').textContent = '✓ ' + file.name;
    dropZone.classList.add('has-file');
    dropZone.classList.remove('dragover');
    if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
      const reader = new FileReader();
      reader.onload = ev => {
        textarea.value = ev.target.result;
        charCount.textContent = ev.target.result.length.toLocaleString() + ' characters';
      };
      reader.readAsText(file);
    } else {
      textarea.value = '';
    }
  }

  // ── Error helpers ─────────────────────────────────────────
  function showError(msg) { uiError.textContent = msg; uiError.classList.remove('hidden'); }
  function hideError()    { uiError.textContent = '';  uiError.classList.add('hidden'); }

  // ── Mode Toggle ───────────────────────────────────────────
  const PIPE_LSTM_STEP  = document.getElementById('pipe-lstm');
  const PIPE_LSTM_ARROW = PIPE_LSTM_STEP ? PIPE_LSTM_STEP.previousElementSibling : null;

  modeToggleBtn.addEventListener('click', () => {
    accurateMode = !accurateMode;
    applyModeUI();
  });

  function applyModeUI() {
    if (accurateMode) {
      // Switch to High-Accuracy mode
      document.body.classList.add('accurate-mode');
      modeToggleBtn.classList.add('active');
      modeToggleBtn.setAttribute('aria-pressed', 'true');
      modeLabel.textContent = 'High-Accuracy Mode';
      document.getElementById('mode-icon-standard').classList.add('hidden');
      document.getElementById('mode-icon-accurate').classList.remove('hidden');

      // Hide LSTM step in pipeline diagram
      if (PIPE_LSTM_STEP)  PIPE_LSTM_STEP.classList.add('hidden-step');
      if (PIPE_LSTM_ARROW) PIPE_LSTM_ARROW.classList.add('hidden-step');

      // Update page header
      document.getElementById('page-title').textContent = 'High-Accuracy Analysis';
      document.getElementById('page-tagline').textContent = 'BART Transformer + BioBERT NER — no LSTM preprocessing';

      // Show banner
      modeBannerContent.innerHTML = `
        <svg class="banner-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>
        </svg>
        <span><strong>High-Accuracy Mode active.</strong> LSTM preprocessing is skipped — BART receives the raw clinical text directly for more faithful summarization.</span>`;
      modeBanner.classList.add('visible');

      // Update button label
      analyzeBtnTx.textContent = 'Run High-Accuracy Analysis';

      // Dim the LSTM result block
      const lstmBlock = document.getElementById('lstm-block');
      if (lstmBlock) lstmBlock.style.opacity = '0.45';

    } else {
      // Switch back to Standard mode
      document.body.classList.remove('accurate-mode');
      modeToggleBtn.classList.remove('active');
      modeToggleBtn.setAttribute('aria-pressed', 'false');
      modeLabel.textContent = 'Standard Mode';
      document.getElementById('mode-icon-standard').classList.remove('hidden');
      document.getElementById('mode-icon-accurate').classList.add('hidden');

      // Restore LSTM step
      if (PIPE_LSTM_STEP)  PIPE_LSTM_STEP.classList.remove('hidden-step');
      if (PIPE_LSTM_ARROW) PIPE_LSTM_ARROW.classList.remove('hidden-step');

      // Restore page header
      document.getElementById('page-title').textContent = 'Clinical Narrative Analysis';
      document.getElementById('page-tagline').textContent = '3-stage AI pipeline: LSTM → BART Transformer → BioBERT NER';

      // Hide banner
      modeBanner.classList.remove('visible');

      // Restore button label
      analyzeBtnTx.textContent = 'Run Analysis Pipeline';

      // Restore LSTM block
      const lstmBlock = document.getElementById('lstm-block');
      if (lstmBlock) lstmBlock.style.opacity = '1';
    }
  }

  // ── Pipeline step animation ───────────────────────────────
  const PIPE_STEPS    = ['pipe-input', 'pipe-lstm', 'pipe-bart', 'pipe-ner'];
  const LOADING_STEPS = ['lstep-1', 'lstep-2', 'lstep-3', 'lstep-4'];
  const STEP_LABELS   = [
    'Initializing pipeline…',
    'LSTM Keyword Extraction…',
    'BART Summarization…',
    'BioBERT NER Analysis…'
  ];
  const ACCURATE_LABELS = [
    'Initializing pipeline…',
    'Skipping LSTM…',
    'BART Summarization…',
    'BioBERT NER Analysis…'
  ];

  let stepTimer = null;
  function startPipelineAnimation() {
    PIPE_STEPS.forEach(id => { const el = document.getElementById(id); if (el) el.classList.remove('active', 'done'); });
    LOADING_STEPS.forEach(id => { const el = document.getElementById(id); if (el) el.classList.remove('active', 'done'); });
    let step = 0;
    const labels = accurateMode ? ACCURATE_LABELS : STEP_LABELS;
    function advance() {
      if (step < PIPE_STEPS.length) {
        if (step > 0) { const prev = document.getElementById(PIPE_STEPS[step - 1]); if (prev) prev.classList.replace('active', 'done'); }
        const cur = document.getElementById(PIPE_STEPS[step]); if (cur) cur.classList.add('active');
      }
      if (step < LOADING_STEPS.length) {
        if (step > 0) { const prev = document.getElementById(LOADING_STEPS[step - 1]); if (prev) { prev.classList.remove('active'); prev.classList.add('done'); } }
        const cur = document.getElementById(LOADING_STEPS[step]); if (cur) cur.classList.add('active');
        const lbl = document.getElementById('loading-label'); if (lbl) lbl.textContent = labels[step] || 'Processing…';
      }
      step++;
      stepTimer = setTimeout(advance, 5500);
    }
    advance();
  }

  function stopPipelineAnimation() {
    clearTimeout(stepTimer);
    PIPE_STEPS.forEach(id => { const el = document.getElementById(id); if (el) { el.classList.remove('active'); el.classList.add('done'); } });
    LOADING_STEPS.forEach(id => { const el = document.getElementById(id); if (el) { el.classList.remove('active'); el.classList.add('done'); } });
  }

  // ── Tab system ────────────────────────────────────────────
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  function switchTab(targetId) {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === targetId);
    });
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.toggle('hidden', p.id !== targetId);
    });
  }

  // ── Analyze ───────────────────────────────────────────────
  analyzeBtn.addEventListener('click', async () => {
    hideError();
    const text   = textarea.value.trim();
    const hasPdf = currentFile && (currentFile.type === 'application/pdf' || currentFile.name.endsWith('.pdf'));

    if (!currentFile && text.length < 20) {
      showError('Please paste a medical report (min 20 characters) or upload a file.');
      return;
    }

    // Enter loading state
    analyzeBtn.disabled   = true;
    analyzeBtnTx.textContent = 'Analyzing…';
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    startPipelineAnimation();

    try {
      const formData = new FormData();
      if (currentFile) formData.append('file', currentFile);
      if (!hasPdf && text) formData.append('text', text);
      formData.append('skip_lstm', accurateMode ? 'true' : 'false');

      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || 'Analysis failed');
      }

      const data = await res.json();
      stopPipelineAnimation();
      renderResults(data);

    } catch (err) {
      stopPipelineAnimation();
      showError('Error: ' + err.message);
      emptyState.classList.remove('hidden');
      loadingState.classList.add('hidden');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtnTx.textContent = accurateMode ? 'Run High-Accuracy Analysis' : 'Run Analysis Pipeline';
    }
  });

  // ── Render Results ────────────────────────────────────────
  function renderResults(data) {
    loadingState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Summaries
    const bartOut = document.getElementById('bart-out');
    const lstmOut = document.getElementById('lstm-out');
    if (bartOut) bartOut.textContent = data.bart_summary || 'No summary generated.';
    if (lstmOut) {
      lstmOut.textContent = data.lstm_summary || 'LSTM not available.';
      lstmOut.className   = 'summary-text ' + (accurateMode ? 'dim-text' : '');
    }

    // Risk card
    const risk = data.risk || { level: 'Low', score: 0, reason: '' };
    const riskCard = document.getElementById('risk-card');
    const riskLabel = document.getElementById('risk-label');
    const riskReason = document.getElementById('risk-reason');
    if (riskCard)   riskCard.className = 'kpi-card risk-kpi ' + risk.level.toLowerCase();
    if (riskLabel)  riskLabel.textContent  = risk.level;
    if (riskReason) riskReason.textContent = risk.reason || '—';

    // Confidence
    const conf    = data.confidence || { overall_pct: 0 };
    const confPct = document.getElementById('conf-pct');
    const confBar = document.getElementById('conf-bar');
    if (confPct) confPct.textContent = conf.overall_pct + '%';
    if (confBar) setTimeout(() => { confBar.style.width = conf.overall_pct + '%'; }, 200);

    // Metadata
    const meta       = data.metadata || {};
    const metaTime   = document.getElementById('meta-time');
    const metaDevice = document.getElementById('meta-device');
    if (metaTime)   metaTime.textContent   = (meta.processing_time_s ?? '—') + 's';
    if (metaDevice) metaDevice.textContent = meta.device === 'cuda:0' ? 'GPU' : 'CPU';

    // Entities
    const ents = data.entities || {};
    populateEntities('disease',   ents.Disease   || []);
    populateEntities('symptom',   ents.Symptom   || []);
    populateEntities('drug',      ents.Drug      || []);
    populateEntities('treatment', ents.Treatment || []);

    const total = (ents.Disease||[]).length + (ents.Symptom||[]).length
                + (ents.Drug||[]).length + (ents.Treatment||[]).length;
    const badge = document.getElementById('entity-count-badge');
    if (badge) badge.textContent = total;

    // Slip
    prepareSlip(data);
    switchTab('tab-summary');
  }

  function populateEntities(cat, items) {
    const ul    = document.getElementById('ent-' + cat);
    const count = document.getElementById('count-' + cat);
    if (!ul) return;
    ul.innerHTML = '';
    if (count) count.textContent = items.length;
    if (!items.length) {
      ul.innerHTML = '<li style="opacity:0.4;background:transparent;border:none;font-size:0.77rem;">None detected</li>';
      return;
    }
    items.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      ul.appendChild(li);
    });
  }

  // ── Prepare Medical Slip ──────────────────────────────────
  function prepareSlip(data) {
    const today = new Date().toLocaleDateString('en-IN', { year:'numeric', month:'long', day:'numeric' });
    const ref   = 'MR-' + Math.floor(Math.random() * 1000000).toString().padStart(6, '0');

    const slipDate = document.getElementById('slip-date');
    const slipRisk = document.getElementById('slip-risk');
    const slipConf = document.getElementById('slip-conf');
    const slipSum  = document.getElementById('slip-summary');
    const slipRef  = document.getElementById('slip-ref');

    if (slipRef)  slipRef.textContent  = 'Reference: ' + ref;
    if (slipDate) slipDate.textContent = today;
    if (slipRisk) slipRisk.textContent = (data.risk?.level ?? '—') + ' Risk';
    if (slipConf) slipConf.textContent = (data.confidence?.overall_pct ?? '—') + '%';
    if (slipSum)  slipSum.textContent  = data.bart_summary || 'No summary available.';

    const ents   = data.entities || {};
    const issues = [...(ents.Disease||[]), ...(ents.Symptom||[])];
    const plans  = [...(ents.Treatment||[]), ...(ents.Drug||[])];

    const slipIssues = document.getElementById('slip-issues');
    const slipPlan   = document.getElementById('slip-plan');
    if (slipIssues) slipIssues.innerHTML = issues.length ? issues.map(i => `<li>${i}</li>`).join('') : '<li>No specific issues detected.</li>';
    if (slipPlan)   slipPlan.innerHTML   = plans.length  ? plans.map(p  => `<li>${p}</li>`).join('') : '<li>No specific action plan detected.</li>';
  }

  // ── PDF Download ──────────────────────────────────────────
  if (dlBtn) {
    dlBtn.addEventListener('click', () => {
      dlBtn.disabled = true;
      dlBtn.textContent = 'Generating PDF…';

      const opt = {
        margin:      [0.4, 0.5],
        filename:    'MediReporter_Clinical_Record.pdf',
        image:       { type: 'jpeg', quality: 0.97 },
        html2canvas: { scale: 2, useCORS: true, backgroundColor: '#ffffff' },
        jsPDF:       { unit: 'in', format: 'a4', orientation: 'portrait' }
      };

      html2pdf().set(opt).from(slipContainer).save()
        .then(() => {
          dlBtn.disabled = false;
          dlBtn.textContent = 'Export Record as PDF';
        })
        .catch(err => {
          console.error('PDF error', err);
          alert('PDF generation failed. Please try again.');
          dlBtn.disabled = false;
          dlBtn.textContent = 'Export Record as PDF';
        });
    });
  }

});
