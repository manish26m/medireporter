/**
 * MediReporter v2.0 — Frontend Application
 * Handles: file upload, API calls, tab switching,
 * pipeline animation, risk/confidence rendering, PDF export.
 */
document.addEventListener('DOMContentLoaded', () => {

  // ── Element refs ────────────────────────────────────────
  const dropZone     = document.getElementById('drop-zone');
  const fileInput    = document.getElementById('file-input');
  const textarea     = document.getElementById('report-text');
  const analyzeBtn   = document.getElementById('analyze-btn');
  const analyzeBtnTx = document.getElementById('analyze-btn-text');
  const charCount    = document.getElementById('char-count');
  const uiError      = document.getElementById('ui-error');

  const emptyState   = document.getElementById('empty-state');
  const loadingState = document.getElementById('loading-state');
  const resultsContent = document.getElementById('results-content');

  const dlBtn        = document.getElementById('download-slip-btn');
  const slipContainer = document.getElementById('medical-slip-container');

  let currentFile = null;
  let lastData    = null;

  // ── Char counter ─────────────────────────────────────────
  textarea.addEventListener('input', () => {
    const n = textarea.value.length;
    charCount.textContent = n.toLocaleString() + ' characters';
  });

  // ── File Upload ───────────────────────────────────────────
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => { if(e.key==='Enter'||e.key===' ') fileInput.click(); });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', e => { if(e.target.files.length) handleFile(e.target.files[0]); });

  function handleFile(file) {
    hideError();
    const ok = file.type === 'text/plain' || file.type === 'application/pdf'
      || file.name.endsWith('.txt') || file.name.endsWith('.pdf');
    if (!ok) { showError('Please upload a .txt or .pdf file.'); return; }
    currentFile = file;
    document.getElementById('upload-text').textContent = '✓ ' + file.name;
    dropZone.style.borderColor = 'var(--teal)';
    if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
      const reader = new FileReader();
      reader.onload = ev => { textarea.value = ev.target.result; charCount.textContent = ev.target.result.length.toLocaleString()+' characters'; };
      reader.readAsText(file);
    } else {
      textarea.value = '';
    }
  }

  // ── Error helpers ─────────────────────────────────────────
  function showError(msg) { uiError.textContent = msg; uiError.classList.remove('hidden'); }
  function hideError()    { uiError.textContent = '';  uiError.classList.add('hidden'); }

  // ── Pipeline step animation ───────────────────────────────
  const PIPE_STEPS   = ['pipe-input','pipe-lstm','pipe-bart','pipe-ner','pipe-report'];
  const LOADING_STEPS = ['lstep-1','lstep-2','lstep-3','lstep-4'];
  const STEP_LABELS  = ['Initializing pipeline…','LSTM Keyword Extraction…','BART Summarization…','BioBERT NER Analysis…','Risk Classification…'];

  let stepTimer = null;
  function startPipelineAnimation() {
    PIPE_STEPS.forEach(id => { const el=document.getElementById(id); if(el){el.classList.remove('active','done');} });
    LOADING_STEPS.forEach(id => { const el=document.getElementById(id); if(el){el.classList.remove('active','done');} });
    let step = 0;
    function advance() {
      if (step < PIPE_STEPS.length) {
        if (step > 0) {
          const prev = document.getElementById(PIPE_STEPS[step-1]);
          if(prev) prev.classList.replace('active','done');
        }
        const cur = document.getElementById(PIPE_STEPS[step]);
        if(cur) cur.classList.add('active');
      }
      if (step < LOADING_STEPS.length) {
        if (step > 0) {
          const prev = document.getElementById(LOADING_STEPS[step-1]);
          if(prev) { prev.classList.remove('active'); prev.classList.add('done'); }
        }
        const cur = document.getElementById(LOADING_STEPS[step]);
        if(cur) cur.classList.add('active');
        document.getElementById('loading-label').textContent = STEP_LABELS[step] || 'Processing…';
      }
      step++;
      stepTimer = setTimeout(advance, 5500);
    }
    advance();
  }

  function stopPipelineAnimation() {
    clearTimeout(stepTimer);
    PIPE_STEPS.forEach(id => { const el=document.getElementById(id); if(el){ el.classList.remove('active'); el.classList.add('done'); } });
    LOADING_STEPS.forEach(id => { const el=document.getElementById(id); if(el){ el.classList.remove('active'); el.classList.add('done'); } });
  }

  // ── Tab system ────────────────────────────────────────────
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  function switchTab(targetId) {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === targetId);
      b.setAttribute('aria-selected', b.dataset.tab === targetId);
    });
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.toggle('hidden', p.id !== targetId);
      if (p.id === targetId) p.classList.remove('hidden');
    });
  }

  // ── Analyze ───────────────────────────────────────────────
  analyzeBtn.addEventListener('click', async () => {
    hideError();
    const text = textarea.value.trim();
    const hasPdf = currentFile && (currentFile.type === 'application/pdf' || currentFile.name.endsWith('.pdf'));

    if (!currentFile && text.length < 20) {
      showError('Please paste a medical report (min 20 characters) or upload a .txt / .pdf file.');
      return;
    }

    // UI: enter loading
    analyzeBtn.disabled = true;
    analyzeBtnTx.textContent = 'Analyzing…';
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    startPipelineAnimation();

    try {
      const formData = new FormData();
      if (currentFile) formData.append('file', currentFile);
      if (!hasPdf && text) formData.append('text', text);

      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || 'Analysis failed');
      }

      const data = await res.json();
      lastData = data;
      stopPipelineAnimation();
      renderResults(data);

    } catch (err) {
      stopPipelineAnimation();
      showError('Error: ' + err.message);
      emptyState.classList.remove('hidden');
      loadingState.classList.add('hidden');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtnTx.textContent = 'Analyze Report';
    }
  });

  // ── Render Results ────────────────────────────────────────
  function renderResults(data) {
    loadingState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Summaries
    document.getElementById('bart-out').textContent = data.bart_summary || 'No summary generated.';
    document.getElementById('lstm-out').textContent = data.lstm_summary || 'LSTM not available.';

    // Risk badge
    const risk  = data.risk || { level:'Low', score:0, reason:'' };
    const badge = document.getElementById('risk-badge');
    badge.className = 'risk-badge ' + risk.level.toLowerCase();
    document.getElementById('risk-label').textContent = risk.level;
    document.getElementById('risk-reason').textContent = risk.reason || '';

    // Confidence bar
    const conf = data.confidence || { overall_pct: 0 };
    document.getElementById('conf-pct').textContent = conf.overall_pct + '%';
    setTimeout(() => {
      document.getElementById('conf-bar').style.width = conf.overall_pct + '%';
    }, 200);

    // Metadata strip
    const meta = data.metadata || {};
    document.getElementById('meta-time').textContent   = '⏱ ' + (meta.processing_time_s ?? '—') + 's';
    document.getElementById('meta-words').textContent  = '📝 ' + (meta.word_count ?? '—') + ' words';
    document.getElementById('meta-device').textContent = '🖥 ' + (meta.device === 'cuda:0' ? 'GPU' : 'CPU');
    document.getElementById('meta-ver').textContent    = 'v' + (meta.pipeline_version ?? '2.0.0');

    // Entities
    const ents = data.entities || {};
    populateEntities('disease',  ents.Disease   || []);
    populateEntities('symptom',  ents.Symptom   || []);
    populateEntities('drug',     ents.Drug      || []);
    populateEntities('treatment',ents.Treatment || []);

    const totalEnts = (ents.Disease||[]).length + (ents.Symptom||[]).length
                    + (ents.Drug||[]).length + (ents.Treatment||[]).length;
    document.getElementById('entity-count-badge').textContent = totalEnts;

    // Prepare slip
    prepareSlip(data);

    // Switch to summary tab
    switchTab('tab-summary');
  }

  function populateEntities(cat, items) {
    const ul    = document.getElementById('ent-' + cat);
    const count = document.getElementById('count-' + cat);
    ul.innerHTML = '';
    count.textContent = items.length;
    if (!items.length) {
      ul.innerHTML = '<li style="opacity:0.4;background:transparent;padding:0;font-size:0.78rem;">None detected</li>';
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
    document.getElementById('slip-date').textContent = today;
    document.getElementById('slip-risk').textContent = (data.risk?.level ?? '—') + ' Risk';
    document.getElementById('slip-conf').textContent = (data.confidence?.overall_pct ?? '—') + '%';
    document.getElementById('slip-summary').textContent = data.bart_summary || 'No summary available.';

    const ents   = data.entities || {};
    const issues = [...(ents.Disease||[]), ...(ents.Symptom||[])];
    const plans  = [...(ents.Treatment||[]), ...(ents.Drug||[])];

    document.getElementById('slip-issues').innerHTML = issues.length
      ? issues.map(i => `<li>${i}</li>`).join('')
      : '<li>No specific issues detected.</li>';
    document.getElementById('slip-plan').innerHTML = plans.length
      ? plans.map(p => `<li>${p}</li>`).join('')
      : '<li>No specific action plan detected.</li>';
  }

  // ── PDF Download ──────────────────────────────────────────
  if (dlBtn) {
    dlBtn.addEventListener('click', () => {
      dlBtn.disabled = true;
      dlBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> Generating…`;

      const opt = {
        margin:      [0.4, 0.5],
        filename:    'MediReporter_Clinical_Slip.pdf',
        image:       { type: 'jpeg', quality: 0.97 },
        html2canvas: { scale: 2, useCORS: true, backgroundColor: '#ffffff' },
        jsPDF:       { unit: 'in', format: 'a4', orientation: 'portrait' }
      };

      html2pdf().set(opt).from(slipContainer).save()
        .then(() => {
          dlBtn.disabled = false;
          dlBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg> Download as PDF`;
        })
        .catch(err => {
          console.error('PDF error', err);
          alert('PDF generation failed. Please try again.');
          dlBtn.disabled = false;
          dlBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg> Download as PDF`;
        });
    });
  }

});
