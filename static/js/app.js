/* ==========================================================================
   MER-Lab Web Studio • Reactive Client Logic
   ========================================================================== */

let currentDatasets = {};
let trainingPollInterval = null;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSliders();
  loadSystemInfo();
  loadDatasets();
  setupEventListeners();

  // Run initial prediction for preview
  setTimeout(runInference, 300);
});

/* Tab Switching */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });
}

/* Slider Live Values */
function initSliders() {
  const epochsSlider = document.getElementById('epochsSlider');
  const epochsVal = document.getElementById('epochsVal');
  epochsSlider.addEventListener('input', () => { epochsVal.textContent = epochsSlider.value; });

  const dropoutSlider = document.getElementById('dropoutSlider');
  const dropoutVal = document.getElementById('dropoutVal');
  dropoutSlider.addEventListener('input', () => { dropoutVal.textContent = dropoutSlider.value; });

  const audioSlider = document.getElementById('audioSlider');
  const audioSliderVal = document.getElementById('audioSliderVal');
  audioSlider.addEventListener('input', () => { audioSliderVal.textContent = audioSlider.value; });

  const videoSlider = document.getElementById('videoSlider');
  const videoSliderVal = document.getElementById('videoSliderVal');
  videoSlider.addEventListener('input', () => { videoSliderVal.textContent = videoSlider.value; });
}

/* Load System Hardware Info */
async function loadSystemInfo() {
  try {
    const res = await fetch('/api/info');
    const data = await res.json();
    const hardwareText = document.getElementById('hardwareText');
    if (data.cuda_available) {
      hardwareText.textContent = `GPU: ${data.gpu_name}`;
    } else {
      hardwareText.textContent = 'Compute: CPU';
    }
  } catch (err) {
    console.warn('Could not fetch hardware info:', err);
  }
}

/* Load Datasets and Presets */
async function loadDatasets() {
  try {
    const res = await fetch('/api/datasets');
    currentDatasets = await res.json();
    updatePresetChips();
  } catch (err) {
    console.warn('Could not fetch datasets:', err);
  }
}

/* Populate Preset Utterance Chips */
function updatePresetChips() {
  const datasetSelect = document.getElementById('datasetSelect');
  const currentKey = datasetSelect.value;
  const container = document.getElementById('presetChips');
  container.innerHTML = '';

  const meta = currentDatasets[currentKey];
  if (!meta || !meta.sample_utterances) return;

  meta.sample_utterances.forEach((sample, i) => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = `"${sample.text.slice(0, 30)}..." (${sample.expected})`;
    chip.addEventListener('click', () => {
      document.getElementById('utteranceInput').value = sample.text;
      document.getElementById('audioSlider').value = sample.audio;
      document.getElementById('audioSliderVal').textContent = sample.audio;
      document.getElementById('videoSlider').value = sample.video;
      document.getElementById('videoSliderVal').textContent = sample.video;
      runInference();
    });
    container.appendChild(chip);
  });
}

/* Setup Event Listeners */
function setupEventListeners() {
  document.getElementById('datasetSelect').addEventListener('change', () => {
    updatePresetChips();
    runInference();
  });

  document.getElementById('runPredictBtn').addEventListener('click', runInference);
  document.getElementById('startTrainBtn').addEventListener('click', () => startTraining(false));
  document.getElementById('quickTestBtn').addEventListener('click', () => startTraining(true));
}

/* Launch Model Training */
async function startTraining(quickTest = false) {
  const dataset = document.getElementById('datasetSelect').value;
  const fusion = document.getElementById('fusionSelect').value;
  const epochs = quickTest ? 1 : parseInt(document.getElementById('epochsSlider').value, 10);
  const lr = parseFloat(document.getElementById('lrSelect').value);
  const dropout = parseFloat(document.getElementById('dropoutSlider').value);
  const numSamples = quickTest ? 100 : parseInt(document.getElementById('samplesSelect').value, 10);
  const useClassWeights = document.getElementById('classWeightCheckbox').checked;

  const startBtn = document.getElementById('startTrainBtn');
  const quickBtn = document.getElementById('quickTestBtn');
  startBtn.disabled = true;
  quickBtn.disabled = true;

  document.getElementById('statusPulse').classList.add('active');
  document.getElementById('statusTitle').textContent = `Training ${dataset.toUpperCase()} (${epochs} Epochs)...`;
  document.getElementById('statusSubtitle').textContent = `Running ${fusion.replace(/_/g, ' ')} on ${numSamples} samples...`;
  document.getElementById('bestEpochBadge').style.display = 'none';

  // Switch to training tab automatically
  document.querySelectorAll('.tab-btn')[0].click();

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: dataset,
        fusion: fusion,
        epochs: epochs,
        batch_size: 32,
        learning_rate: lr,
        modality_dropout: dropout,
        use_class_weights: useClassWeights,
        num_samples: numSamples,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start training');
    }

    // Start polling
    if (trainingPollInterval) clearInterval(trainingPollInterval);
    trainingPollInterval = setInterval(pollTrainingStatus, 800);

  } catch (err) {
    alert(`Error: ${err.message}`);
    startBtn.disabled = false;
    quickBtn.disabled = false;
    document.getElementById('statusPulse').classList.remove('active');
    document.getElementById('statusTitle').textContent = 'Training Error';
    document.getElementById('statusSubtitle').textContent = err.message;
  }
}

/* Poll Training Status */
async function pollTrainingStatus() {
  try {
    const res = await fetch('/api/train/status');
    const state = await res.json();

    document.getElementById('curEpochMetric').textContent = `${state.current_epoch} / ${state.total_epochs}`;

    if (state.val_loss.length > 0) {
      const latestValLoss = state.val_loss[state.val_loss.length - 1];
      const latestValAcc = state.val_acc[state.val_acc.length - 1];
      const latestValF1 = state.val_weighted_f1[state.val_weighted_f1.length - 1];

      document.getElementById('valF1Metric').textContent = `${(latestValF1 * 100).toFixed(1)}%`;
      document.getElementById('valAccMetric').textContent = `${(latestValAcc * 100).toFixed(1)}%`;
      document.getElementById('bestF1Metric').textContent = `${(state.best_score * 100).toFixed(1)}%`;

      // Update Charts
      drawLossChart(state.train_loss, state.val_loss);
      drawF1Chart(state.val_weighted_f1, state.best_epoch);

      // Update History Table
      renderHistoryTable(state.history);
    }

    if (!state.is_training) {
      clearInterval(trainingPollInterval);
      document.getElementById('statusPulse').classList.remove('active');
      document.getElementById('startTrainBtn').disabled = false;
      document.getElementById('quickTestBtn').disabled = false;

      document.getElementById('statusTitle').textContent = state.error ? 'Training Failed' : 'Training Complete!';
      document.getElementById('statusSubtitle').textContent = state.status;

      if (state.best_epoch) {
        document.getElementById('bestEpochNum').textContent = state.best_epoch;
        document.getElementById('bestEpochBadge').style.display = 'inline-flex';
      }
    }
  } catch (err) {
    console.error('Error polling status:', err);
  }
}

/* Render Epoch History Table */
function renderHistoryTable(history) {
  const tbody = document.getElementById('historyTableBody');
  if (!history || history.length === 0) return;

  tbody.innerHTML = '';
  history.forEach(item => {
    const row = document.createElement('tr');
    row.style.borderBottom = '1px solid rgba(255, 255, 255, 0.04)';

    row.innerHTML = `
      <td style="padding: 0.6rem 0.75rem; font-weight: 700;">Epoch ${item.epoch}</td>
      <td style="padding: 0.6rem 0.75rem; color: #94a3b8;">${item.train_loss.toFixed(4)}</td>
      <td style="padding: 0.6rem 0.75rem; color: #f43f5e;">${item.val_loss.toFixed(4)}</td>
      <td style="padding: 0.6rem 0.75rem; color: #10b981;">${(item.val_acc * 100).toFixed(1)}%</td>
      <td style="padding: 0.6rem 0.75rem; color: #06b6d4; font-weight: 600;">${(item.val_weighted_f1 * 100).toFixed(1)}%</td>
      <td style="padding: 0.6rem 0.75rem;">
        ${item.is_best ? '<span class="badge" style="border-color: rgba(245, 158, 11, 0.4); color: #fbbf24;">⭐ Best Checkpoint</span>' : '<span style="color: #64748b;">Recorded</span>'}
      </td>
    `;
    tbody.appendChild(row);
  });
}

/* Draw Loss Trajectory Canvas Chart */
function drawLossChart(trainLoss, valLoss) {
  const canvas = document.getElementById('lossCanvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 240 * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = 240;
  ctx.clearRect(0, 0, w, h);

  if (!trainLoss || trainLoss.length === 0) return;

  const maxVal = Math.max(...trainLoss, ...valLoss, 2.0);
  const minVal = Math.min(...trainLoss, ...valLoss, 0.0);
  const padLeft = 40, padRight = 20, padTop = 20, padBottom = 30;
  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();

    const val = maxVal - ((maxVal - minVal) / 4) * i;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter';
    ctx.fillText(val.toFixed(2), 8, y + 3);
  }

  function plotSeries(data, color) {
    if (data.length < 1) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    data.forEach((val, idx) => {
      const x = padLeft + (chartW / Math.max(data.length - 1, 1)) * idx;
      const y = padTop + chartH - ((val - minVal) / Math.max(maxVal - minVal, 0.01)) * chartH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw points
    ctx.fillStyle = color;
    data.forEach((val, idx) => {
      const x = padLeft + (chartW / Math.max(data.length - 1, 1)) * idx;
      const y = padTop + chartH - ((val - minVal) / Math.max(maxVal - minVal, 0.01)) * chartH;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  plotSeries(trainLoss, '#6366f1'); // Train Loss (Indigo)
  plotSeries(valLoss, '#f43f5e');   // Val Loss (Rose)

  // Legend
  ctx.fillStyle = '#6366f1';
  ctx.fillText('● Train Loss', padLeft + 10, h - 8);
  ctx.fillStyle = '#f43f5e';
  ctx.fillText('● Validation Loss', padLeft + 100, h - 8);
}

/* Draw Weighted F1 Canvas Chart */
function drawF1Chart(f1Scores, bestEpoch) {
  const canvas = document.getElementById('f1Canvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 240 * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = 240;
  ctx.clearRect(0, 0, w, h);

  if (!f1Scores || f1Scores.length === 0) return;

  const padLeft = 40, padRight = 20, padTop = 20, padBottom = 30;
  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();

    const val = 1.0 - 0.25 * i;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter';
    ctx.fillText(`${(val * 100).toFixed(0)}%`, 8, y + 3);
  }

  // Plot F1 Line
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 3;
  ctx.beginPath();
  f1Scores.forEach((val, idx) => {
    const x = padLeft + (chartW / Math.max(f1Scores.length - 1, 1)) * idx;
    const y = padTop + chartH - (val / 1.0) * chartH;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw points with star on best epoch
  f1Scores.forEach((val, idx) => {
    const x = padLeft + (chartW / Math.max(f1Scores.length - 1, 1)) * idx;
    const y = padTop + chartH - (val / 1.0) * chartH;
    const isBest = (idx + 1) === bestEpoch;

    ctx.fillStyle = isBest ? '#f59e0b' : '#06b6d4';
    ctx.beginPath();
    ctx.arc(x, y, isBest ? 7 : 4, 0, Math.PI * 2);
    ctx.fill();

    if (isBest) {
      ctx.fillStyle = '#f59e0b';
      ctx.font = 'bold 11px Outfit';
      ctx.fillText('⭐ Best', x - 15, y - 12);
    }
  });

  ctx.fillStyle = '#06b6d4';
  ctx.fillText('● Validation Weighted-F1 Score', padLeft + 10, h - 8);
}

/* Run Interactive Inference Playground */
async function runInference() {
  const dataset = document.getElementById('datasetSelect').value;
  const utterance = document.getElementById('utteranceInput').value;
  const audioVal = parseFloat(document.getElementById('audioSlider').value);
  const videoVal = parseFloat(document.getElementById('videoSlider').value);

  const droppedMods = [];
  if (document.getElementById('maskText').checked) droppedMods.push('text');
  if (document.getElementById('maskAudio').checked) droppedMods.push('audio');
  if (document.getElementById('maskVideo').checked) droppedMods.push('video');

  const predictBtn = document.getElementById('runPredictBtn');
  predictBtn.disabled = true;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset: dataset,
        text_utterance: utterance,
        audio_pitch_energy: audioVal,
        visual_affect_intensity: videoVal,
        modality_mask: droppedMods,
      }),
    });

    const data = await res.json();

    // Update Top Prediction
    document.getElementById('predEmotion').textContent = data.predicted_emotion;
    document.getElementById('predConfidence').textContent = `${data.confidence.toFixed(1)}%`;

    // Update Dynamic Gating (α)
    const gw = data.gating_weights;
    document.getElementById('gateText').textContent = gw.text.toFixed(2);
    document.getElementById('gateAudio').textContent = gw.audio.toFixed(2);
    document.getElementById('gateVideo').textContent = gw.video.toFixed(2);

    document.getElementById('gateBarText').style.width = `${gw.text * 100}%`;
    document.getElementById('gateBarAudio').style.width = `${gw.audio * 100}%`;
    document.getElementById('gateBarVideo').style.width = `${gw.video * 100}%`;

    // Render Probabilities List
    const listContainer = document.getElementById('probList');
    listContainer.innerHTML = '';

    data.class_probabilities.forEach(item => {
      const row = document.createElement('div');
      row.className = `prob-row ${item.is_top ? 'top' : ''}`;
      row.innerHTML = `
        <div class="prob-header">
          <span class="prob-name">${item.emotion} ${item.is_top ? '⭐' : ''}</span>
          <span class="prob-pct">${item.percentage.toFixed(1)}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width: ${item.percentage}%;"></div>
        </div>
      `;
      listContainer.appendChild(row);
    });

  } catch (err) {
    console.error('Error running inference:', err);
  } finally {
    predictBtn.disabled = false;
  }
}
