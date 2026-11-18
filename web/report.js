const form = document.querySelector('#upload-form');
const imageInput = document.querySelector('#image');
const preview = document.querySelector('#preview');
const emptyImage = document.querySelector('#empty-image');
const generate = document.querySelector('#generate');
const statusMessage = document.querySelector('#status');
const placeholder = document.querySelector('#report-placeholder');
const report = document.querySelector('#report-content');
const evidencePanel = document.querySelector('#evidence-panel');
const evidenceRows = document.querySelector('#evidence-rows');
const heatmapPanel = document.querySelector('#heatmap-panel');
const heatmapFrame = document.querySelector('#heatmap-frame');
const heatmapImage = document.querySelector('#heatmap');
const heatmapLegend = document.querySelector('#heatmap-legend');
const chartsPanel = document.querySelector('#charts-panel');
const flagSummary = document.querySelector('#flag-summary');
const flagList = document.querySelector('#flag-list');
const findingSelect = document.querySelector('#finding-select');
const heatmapControls = document.querySelector('#heatmap-controls');
const opacityInput = document.querySelector('#overlay-opacity');
const processedImage = document.querySelector('#processed-image');
const formats = new Set(['image/png', 'image/jpeg']);
const uploadLimit = 10 * 1024 * 1024;
let previewUrl;
let currentAnalysis;
let selectedExample;
let exampleLoading = false;
let currentVerifierCheck;
const verifierPanel = document.querySelector('#verifier-panel');
const resetButton = document.querySelector('#reset-analysis');
const uploadDrop = document.querySelector('#upload-drop');
const displayName = (name) => name.replaceAll('_', ' ');
const percent = (value) => `${(value * 100).toFixed(1)}%`;

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle('error', isError);
}

function setWorkflow(stage) {
  const steps = ['upload', 'analyze', 'review'];
  const current = steps.indexOf(stage);
  document.querySelectorAll('.workflow li').forEach((item, index) => {
    item.classList.toggle('is-active', index === current);
    item.classList.toggle('is-complete', index < current);
    if (index === current) item.setAttribute('aria-current', 'step');
    else item.removeAttribute('aria-current');
  });
}

function clearReport() {
  setWorkflow('upload');
  currentAnalysis = undefined;
  currentVerifierCheck = undefined;
  report.hidden = true;
  placeholder.hidden = false;
  evidencePanel.hidden = true;
  evidenceRows.replaceChildren();
  heatmapPanel.hidden = true;
  heatmapFrame.hidden = true;
  heatmapLegend.hidden = true;
  heatmapImage.removeAttribute('src');
  processedImage.removeAttribute('src');
  chartsPanel.hidden = true;
  flagSummary.hidden = true;
  flagList.replaceChildren();
  findingSelect.replaceChildren();
  heatmapControls.hidden = true;
  verifierPanel.hidden = true;
  document.querySelector('#verifier-result').hidden = true;
  document.querySelector('#verify-status').textContent = '';
  document.querySelector('#challenge-text').value = document.querySelector('#challenge-text').defaultValue;
  for (const id of ['#findings', '#impression', '#corrected-findings', '#corrected-impression', '#report-meta', '#verification-summary']) document.querySelector(id).textContent = '';
  document.querySelector('#correction-log').replaceChildren();
  document.querySelector('#verifier-issues').replaceChildren();
}

function svgElement(tag, attributes = {}, text) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderChart(id, rows, uncertainty = false) {
  const svg = document.querySelector(id);
  svg.querySelectorAll(':scope > g').forEach((group) => group.remove());
  const group = svgElement('g');
  const x = 190, width = 358, top = 61, step = 35;
  const maximum = uncertainty ? Math.max(2, Math.ceil(Math.max(...rows.map((row) => row.std * 100)) / 2) * 2) : 100;
  for (let tick = 0; tick <= 4; tick++) {
    const value = maximum * tick / 4;
    const left = x + width * tick / 4;
    group.append(svgElement('line', {x1:left, x2:left, y1:39, y2:top + step * 13 + 19, class:'chart-gridline'}));
    group.append(svgElement('text', {x:left, y:25, 'text-anchor':'middle', class:'chart-tick'}, `${Number(value.toFixed(1))}${uncertainty ? '' : '%'}`));
  }
  rows.forEach((row, index) => {
    const y = top + index * step;
    const value = (uncertainty ? row.std : row.prob) * 100;
    const bar = svgElement('g');
    bar.append(svgElement('title', {}, `${displayName(row.finding)}: ${uncertainty ? `${value.toFixed(1)} percentage points SD` : `${percent(row.prob)}; reporting threshold ${percent(row.report_threshold)}`}`));
    bar.append(svgElement('text', {x:x-15, y:y+5, 'text-anchor':'end', class:'chart-label'}, displayName(row.finding)));
    bar.append(svgElement('rect', {x, y:y-9, width, height:18, rx:3, class:'chart-track'}));
    bar.append(svgElement('rect', {x, y:y-9, width:Math.max(0, Math.min(width, value/maximum*width)), height:18, rx:3,
      class:uncertainty ? 'chart-variability' : (row.tier === 'absent' ? 'chart-below' : 'chart-flag')}));
    if (!uncertainty) {
      const thresholdX = x + Math.min(1, Math.max(0, row.report_threshold)) * width;
      bar.append(svgElement('line', {x1:thresholdX, x2:thresholdX, y1:y-13, y2:y+13, class:'chart-threshold'}));
    }
    bar.append(svgElement('text', {x:x+width+12, y:y+5, class:'chart-value'}, `${value.toFixed(1)}${uncertainty ? '' : '%'}`));
    group.append(bar);
  });
  group.append(svgElement('text', {x, y:top + step * 14 + 12, class:'chart-tick'}, uncertainty ? 'Standard deviation (percentage points)' : 'Calibrated model score (%)'));
  svg.append(group);
}

function updateHeatmap() {
  if (!currentAnalysis) return;
  const value = findingSelect.value;
  const combined = value === '__combined__';
  heatmapImage.src = combined ? currentAnalysis.heatmap : currentAnalysis.finding_heatmaps[value];
  heatmapImage.alt = `${combined ? 'Combined' : displayName(value)} model activation on the processed X-ray`;
  const selected = currentAnalysis.evidence.find((row) => row.finding === value);
  document.querySelector('#heatmap-caption').textContent = selected ?
    `${displayName(value)} · ${percent(selected.prob)} model score · ${selected.region || 'Location unavailable'} (estimated)` :
    `Combined activation for ${currentAnalysis.heatmap_findings.map(displayName).join(', ')}.`;
  for (const button of flagList.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.dataset.finding === value));
  }
}

function renderAnalysis(data) {
  currentAnalysis = data;
  setWorkflow('review');
  verifierPanel.hidden = false;
  document.querySelector('#report-meta').textContent = `${data.model_run} · ${data.elapsed_seconds.toFixed(1)} s · ${new Date(data.generated_at).toLocaleString()}`;
  document.querySelector('#verification-summary').textContent = data.verification.status === 'consistent' ?
    'Report consistency: no rule errors found against the model evidence. This does not establish clinical correctness.' :
    'Report consistency: rule violations remain. Review the evidence before using this draft.';
  const flagged = data.evidence.filter((row) => row.tier !== 'absent');
  document.querySelector('#flag-heading').textContent = flagged.length ?
    `${flagged.length} ${flagged.length === 1 ? 'finding' : 'findings'} to review` : 'Assessment inconclusive';
  document.querySelector('#flag-description').textContent = flagged.length ?
    'These findings crossed the model’s reporting thresholds. Select one to explore its activation map.' :
    'All scores are below their reporting thresholds. This does not establish a normal X-ray.';
  for (const row of flagged) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'flag-button';
    button.dataset.finding = row.finding;
    button.dataset.tier = row.tier;
    button.setAttribute('aria-pressed', 'false');
    const name = document.createElement('strong');
    name.textContent = displayName(row.finding);
    const score = document.createElement('span');
    score.textContent = percent(row.prob);
    const detail = document.createElement('small');
    detail.textContent = `${row.tier === 'possible' ? 'Possible finding' : 'Above assertion threshold'} · ${row.region || 'Location unavailable'} (estimated)`;
    button.append(name, score, detail);
    button.addEventListener('click', () => { findingSelect.value = row.finding; updateHeatmap(); });
    flagList.append(button);
  }
  flagSummary.hidden = false;
  renderChart('#scores-chart', data.evidence);
  renderChart('#uncertainty-chart', data.evidence, true);
  chartsPanel.hidden = false;
  if (data.heatmap && data.processed_image && data.finding_heatmaps) {
    processedImage.src = data.processed_image;
    if (flagged.length > 1) findingSelect.add(new Option('All flagged findings', '__combined__'));
    for (const row of flagged) findingSelect.add(new Option(displayName(row.finding), row.finding));
    heatmapControls.hidden = false;
    heatmapFrame.hidden = false;
    heatmapLegend.hidden = false;
    opacityInput.value = '100';
    heatmapImage.style.opacity = '1';
    document.querySelector('#opacity-value').value = '100%';
    updateHeatmap();
  }
}

findingSelect.addEventListener('change', updateHeatmap);
opacityInput.addEventListener('input', () => {
  heatmapImage.style.opacity = Number(opacityInput.value) / 100;
  document.querySelector('#opacity-value').value = `${opacityInput.value}%`;
});

function renderEvidence(data) {
  const percent = (value) => `${(value * 100).toFixed(1)}%`;
  for (const row of data.evidence) {
    const tr = document.createElement('tr');
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = row.finding.replaceAll('_', ' ');
    tr.append(name);

    const probability = document.createElement('td');
    probability.textContent = percent(row.prob);
    const meter = document.createElement('span');
    meter.className = 'probability-track';
    meter.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span');
    fill.style.width = percent(row.prob);
    meter.append(fill);
    probability.append(meter);
    tr.append(probability);

    for (const value of [`${(row.std * 100).toFixed(1)} pp`, percent(row.report_threshold)]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      tr.append(cell);
    }

    const statusCell = document.createElement('td');
    const statusLabel = document.createElement('span');
    statusLabel.className = `finding-status ${row.tier}`;
    statusLabel.textContent = row.tier === 'absent' ? 'Below threshold' :
      (row.tier === 'possible' ? 'Possible' : 'Above assertion threshold');
    statusCell.append(statusLabel);
    tr.append(statusCell);
    const location = document.createElement('td');
    location.textContent = row.region || '—';
    tr.append(location);
    evidenceRows.append(tr);
  }
  document.querySelector('#model-run').textContent = `Model run ${data.model_run}`;
  document.querySelector('#uncertainty-note').textContent =
    `Uncertainty is the standard deviation across ${data.uncertainty_samples} dropout samples, in percentage points (pp). Identical decoded images use the same sampling seed for repeatable results in this runtime.`;
  evidencePanel.hidden = false;
  heatmapPanel.hidden = false;

  const caption = document.querySelector('#heatmap-caption');
  if (data.heatmap) {
    heatmapImage.src = data.heatmap;
    heatmapFrame.hidden = false;
    heatmapLegend.hidden = false;
    caption.textContent = `Combined activation for ${data.heatmap_findings.map((name) => name.replaceAll('_', ' ')).join(', ')}.`;
  } else {
    caption.textContent = 'All finding probabilities are below their reporting thresholds.';
  }
}

function prepareFile(file, example) {
  clearReport();
  selectedExample = example;
  if (!example) document.querySelector('#demo-status').textContent = '';
  document.querySelector('#reference-panel').hidden = !example;
  if (example) {
    document.querySelector('#reference-labels').textContent = `Dataset labels: ${example.dataset_labels.map(displayName).join(', ')}.`;
    document.querySelector('#reference-context').textContent = `Source image: ${example.source_image}. ${example.reference_note} Experiment split: ${example.experiment_split}. These selected cases do not establish benchmark performance.`;
  }
  setStatus('');
  preview.hidden = true;
  emptyImage.hidden = false;
  preview.removeAttribute('src');
  uploadDrop.classList.remove('has-image', 'is-dragging');
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = undefined;

  if (!file) return;
  const error = !formats.has(file.type) ? 'Choose a PNG or JPEG image.' :
    (file.size > uploadLimit ? 'Choose an image no larger than 10 MB.' : '');
  if (error) {
    imageInput.value = '';
    setStatus(error, true);
    return;
  }
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  setWorkflow('analyze');
}

imageInput.addEventListener('change', () => prepareFile(imageInput.files[0]));

uploadDrop.addEventListener('click', () => { if (!generate.disabled && !exampleLoading) imageInput.click(); });
uploadDrop.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    if (!generate.disabled && !exampleLoading) imageInput.click();
  }
});
uploadDrop.addEventListener('dragover', (event) => {
  event.preventDefault();
  if (!generate.disabled && !exampleLoading) uploadDrop.classList.add('is-dragging');
});
uploadDrop.addEventListener('dragleave', () => uploadDrop.classList.remove('is-dragging'));
uploadDrop.addEventListener('drop', (event) => {
  event.preventDefault(); uploadDrop.classList.remove('is-dragging');
  if (generate.disabled || exampleLoading) return;
  if (event.dataTransfer.files.length !== 1) { setStatus('Choose one X-ray image at a time.', true); return; }
  const file = event.dataTransfer.files[0];
  const transfer = new DataTransfer(); transfer.items.add(file); imageInput.files = transfer.files;
  prepareFile(file);
});

resetButton.addEventListener('click', () => {
  imageInput.value = '';
  prepareFile(undefined);
  document.querySelector('#demo-status').textContent = '';
});

async function loadExamples() {
  try {
    const [exampleResponse, statusResponse] = await Promise.all([fetch('/api/examples'), fetch('/api/status')]);
    if (!exampleResponse.ok || !statusResponse.ok) throw new Error('Availability check failed');
    const catalog = await exampleResponse.json();
    const status = await statusResponse.json();
    const modelStatus = document.querySelector('#model-status');
    modelStatus.textContent = status.files_ready ? 'Available' : 'Unavailable';
    modelStatus.dataset.state = status.files_ready ? 'available' : 'unavailable';
    modelStatus.title = status.files_ready ? 'Trained checkpoint and calibration files are available.' : 'Model files are missing.';
    const modelNames = {densenet121:'DenseNet-121', densenet121_chexpert:'DenseNet-121 CheXpert', resnet50:'ResNet-50', efficientnet_v2_s:'EfficientNetV2-S', convnext_tiny:'ConvNeXt-Tiny'};
    document.querySelector('#model-backbone').textContent = modelNames[status.backbone] || status.backbone || 'Model not specified';
    document.querySelector('#model-run-label').textContent = status.model_run ? `Run · ${status.model_run}` : 'No model run selected';
    document.querySelector('#model-finding-count').textContent = `${status.findings_count || 14} findings`;
    document.querySelector('#model-sample-count').textContent = status.uncertainty_samples ? `${status.uncertainty_samples} samples` : '—';
    for (const example of catalog.examples) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'demo-button';
      button.textContent = example.title;
      button.title = example.description;
      button.addEventListener('click', async () => {
        if (generate.disabled || exampleLoading) return;
        exampleLoading = true;
        try {
          document.querySelectorAll('.demo-button').forEach((item) => { item.disabled = true; });
          imageInput.disabled = true; resetButton.disabled = true;
          document.querySelector('#demo-status').textContent = 'Loading NIH example…';
          const response = await fetch(`/api/examples/${encodeURIComponent(example.id)}/image`);
          if (!response.ok) throw new Error('The example image is unavailable.');
          const file = new File([await response.blob()], `nih-example-${example.id}.png`, {type:'image/png'});
          const transfer = new DataTransfer(); transfer.items.add(file); imageInput.files = transfer.files;
          prepareFile(file, example);
          document.querySelector('#demo-status').textContent = example.description;
          imageInput.disabled = false;
          form.requestSubmit();
        } catch (error) {
          document.querySelector('#demo-status').textContent = error.message;
        } finally {
          exampleLoading = false;
          if (!generate.disabled) {
            imageInput.disabled = false; resetButton.disabled = false;
            document.querySelectorAll('.demo-button').forEach((item) => { item.disabled = false; });
          }
        }
      });
      document.querySelector('#demo-examples').append(button);
    }
  } catch {
    document.querySelector('#model-status').textContent = 'Unavailable';
    document.querySelector('#model-status').dataset.state = 'unavailable';
    document.querySelector('#model-backbone').textContent = 'Connection unavailable';
    document.querySelector('#model-run-label').textContent = 'Check that the Python server is running';
    document.querySelector('#demo-status').textContent = 'Examples are unavailable. You can still upload an image.';
  }
}

function saveFile(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const link = document.createElement('a'); link.href = url; link.download = name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportRecord() {
  const data = currentAnalysis;
  return {project:'XRayGuard', purpose:'Research demonstration; not a clinical diagnosis', model_run:data.model_run,
    generated_at:data.generated_at, generator:data.generator, findings:data.findings, impression:data.impression,
    verification:data.verification, uncertainty_samples:data.uncertainty_samples, evidence:data.evidence,
    sampling_seed:data.sampling_seed,
    example_reference:selectedExample || null};
}

document.querySelector('#download-report').addEventListener('click', () => {
  if (!currentAnalysis) return;
  const r = exportRecord();
  const lines = ['XRAYGUARD RESEARCH DRAFT', r.purpose, '', `Model: ${r.model_run}`, `Generated: ${r.generated_at}`,
    '', 'FINDINGS', r.findings, '', 'IMPRESSION', r.impression, '', r.verification.scope, '', 'MODEL EVIDENCE'];
  for (const row of r.evidence) lines.push(`${displayName(row.finding)}: score ${percent(row.prob)}, threshold ${percent(row.report_threshold)}, SD ${(row.std*100).toFixed(1)} pp, ${row.tier === 'absent' ? 'below threshold' : row.tier}`);
  saveFile('xrayguard-draft-report.txt', lines.join('\n'), 'text/plain');
});
document.querySelector('#download-evidence').addEventListener('click', () => {
  if (currentAnalysis) saveFile('xrayguard-evidence.json', JSON.stringify(exportRecord(), null, 2), 'application/json');
});
document.querySelector('#print-report').addEventListener('click', () => { if (currentAnalysis) window.print(); });
document.querySelector('#download-trace').addEventListener('click', () => {
  if (currentVerifierCheck) saveFile('xrayguard-verification-trace.json', JSON.stringify(currentVerifierCheck,null,2), 'application/json');
});

document.querySelector('#verify-draft').addEventListener('click', async () => {
  if (!currentAnalysis) return;
  const analysis = currentAnalysis;
  const button = document.querySelector('#verify-draft'); button.disabled = true;
  const status = document.querySelector('#verify-status'); status.textContent = 'Checking wording…';
  document.querySelector('#verifier-result').hidden = true;
  try {
    const response = await fetch('/api/verify', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({findings:document.querySelector('#challenge-text').value, impression:'',
        evidence:Object.fromEntries(analysis.evidence.map((row) => [row.finding, row]))})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The wording check failed.');
    if (analysis !== currentAnalysis) return;
    const list = document.querySelector('#verifier-issues'); list.replaceChildren();
    for (const issue of result.issues) {
      const item = document.createElement('li');
      item.textContent = `${issue.type.replaceAll('_',' ')}: ${issue.detail}`; list.append(item);
    }
    if (!result.issues.length) { const item=document.createElement('li'); item.textContent='No rule errors found against this model evidence.'; list.append(item); }
    document.querySelector('#corrected-findings').textContent = result.findings;
    document.querySelector('#corrected-impression').textContent = result.impression;
    const log = document.querySelector('#correction-log'); log.replaceChildren();
    for (const correction of result.corrections) { const item=document.createElement('li'); item.textContent=correction; log.append(item); }
    if (!result.corrections.length) { const item=document.createElement('li');item.textContent='No changes were required.';log.append(item); }
    currentVerifierCheck = {model_run:analysis.model_run, generated_at:new Date().toISOString(), sampling_seed:analysis.sampling_seed, example_reference:selectedExample || null,
      draft:document.querySelector('#challenge-text').value, evidence:analysis.evidence, ...result};
    status.textContent = `${result.issues.length} rule errors found. ${result.remaining.length ? 'Some violations remain after correction.' : 'No rule errors remain after correction.'}`;
    document.querySelector('#verifier-result').hidden = false;
  } catch (error) { if (analysis === currentAnalysis) status.textContent = error.message; }
  finally { button.disabled = false; }
});

loadExamples();

preview.addEventListener('load', () => {
  preview.hidden = false;
  emptyImage.hidden = true;
  uploadDrop.classList.add('has-image');
});

preview.addEventListener('error', () => {
  imageInput.value = '';
  uploadDrop.classList.remove('has-image');
  preview.hidden = true; emptyImage.hidden = false;
  setWorkflow('upload');
  setStatus('This image could not be opened.', true);
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file || generate.disabled) return;

  clearReport();
  generate.disabled = true;
  generate.dataset.loading = 'true';
  generate.textContent = 'Analyzing image…';
  form.setAttribute('aria-busy', 'true');
  uploadDrop.setAttribute('aria-disabled', 'true');
  setWorkflow('analyze');
  imageInput.disabled = true;
  resetButton.disabled = true;
  document.querySelectorAll('.demo-button').forEach((button) => { button.disabled = true; });
  setStatus('Analyzing the image…');
  try {
    const response = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': file.type },
      body: file,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Report generation failed.');
    if (!Array.isArray(data.evidence) || !data.finding_heatmaps || !data.processed_image) {
      throw new Error('Restart the Python server to load the interactive analysis view.');
    }

    document.querySelector('#findings').textContent = data.findings || 'No findings text generated.';
    document.querySelector('#impression').textContent = data.impression || 'No impression text generated.';
    placeholder.hidden = true;
    report.hidden = false;
    renderEvidence(data);
    renderAnalysis(data);
    setStatus('Draft report ready.');
  } catch (error) {
    setWorkflow('analyze');
    setStatus(error instanceof TypeError ? 'Could not reach the report server.' : error.message, true);
  } finally {
    generate.disabled = false;
    delete generate.dataset.loading;
    generate.textContent = 'Generate report';
    form.setAttribute('aria-busy', 'false');
    uploadDrop.setAttribute('aria-disabled', 'false');
    imageInput.disabled = false;
    resetButton.disabled = false;
    document.querySelectorAll('.demo-button').forEach((button) => { button.disabled = false; });
  }
});
