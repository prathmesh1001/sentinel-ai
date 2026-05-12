/**
 * SENTINEL v3 — app.js
 * Clean redesign: metrics shown once, chatbot in side panel, no layout shift
 */

/* ══════════════════════════════════════════════
   CONSTANTS
══════════════════════════════════════════════ */

const DEVICES = {
  rpi: { id:'RPi-4B-001', model:'RPi 4 Model B', os:'Raspbian 11' },
  jetson: { id:'JSN-NX-001', model:'Jetson Nano 4GB', os:'JetPack 5.1' },
};

// Only 4 cards in the top row (split cleanly)
const METRIC_DEFS = [
  { key:'cpu_temp', label:'CPU TEMP',  unit:'°C', sub:'PROCESSOR CORE',  max:90,  warn:70, fail:82 },
  { key:'gpu_temp', label:'GPU TEMP',  unit:'°C', sub:'GRAPHICS CORE',   max:90,  warn:72, fail:84 },
  { key:'cpu_use',  label:'CPU USAGE', unit:'%',  sub:'ALL CORES AVG',   max:100, warn:75, fail:92 },
  { key:'gpu_use',  label:'GPU USAGE', unit:'%',  sub:'CUDA / OpenCL',   max:100, warn:80, fail:95 },
];

// Additional metrics shown in bottom info only (no card duplication)
const EXTRA_DEFS = [
  { key:'ram',  label:'RAM USAGE', unit:'%', sub:'LPDDR4 / 4 GB', max:100, warn:80, fail:93 },
  { key:'pwr',  label:'PWR CONS.', unit:'W', sub:'BOARD TOTAL',   max:15,  warn:8,  fail:12 },
  { key:'volt', label:'VOLTAGE',   unit:'V', sub:'CORE RAIL',
    max:{rpi:5.5,jetson:9.0}, warn:{rpi:4.9,jetson:7.2}, fail:{rpi:4.7,jetson:7.0}, invert:true },
];

const ALL_DEFS = [...METRIC_DEFS, ...EXTRA_DEFS];
const MAX_HISTORY = 40;

/* ══════════════════════════════════════════════
   STATE
══════════════════════════════════════════════ */

let currentDevice = 'rpi';
let metrics       = {};
let tickCount     = 0;
let uptimeSeconds = 302622;
let tempChart, usageChart;
let metricsBuilt      = false;
let lastMLPred        = null;
let lastMLConfidence  = 0;
let lastMLFailureRisk = 0;

const chatHistory = [];
const MAX_CHAT_HISTORY = 20;

/* ══════════════════════════════════════════════
   HELPERS
══════════════════════════════════════════════ */

function getStatus(def, val) {
  const warn = typeof def.warn === 'object' ? def.warn[currentDevice] : def.warn;
  const fail = typeof def.fail === 'object' ? def.fail[currentDevice] : def.fail;
  if (def.invert) return val < fail ? 'fail' : val < warn ? 'warn' : 'ok';
  return val > fail ? 'fail' : val > warn ? 'warn' : 'ok';
}

function getOverallStatus(m) {
  if (!m || Object.keys(m).length === 0) return 'ok';
  let worst = 'ok';
  for (const def of ALL_DEFS) {
    const s = getStatus(def, m[def.key] || 0);
    if (s === 'fail') return 'fail';
    if (s === 'warn') worst = 'warn';
  }
  return worst;
}

function barPct(def, val) {
  const fail = typeof def.fail === 'object' ? def.fail[currentDevice] : def.fail;
  const max  = typeof def.max  === 'object' ? def.max[currentDevice]  : def.max;
  if (def.invert) {
    const range = max - fail;
    return Math.max(0, Math.min(100, ((val - fail) / range) * 100));
  }
  return Math.max(0, Math.min(100, (val / max) * 100));
}

function formatUptime(s) {
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${d}d ${h}h ${m}m`;
}

/* ══════════════════════════════════════════════
   DATA FETCH
══════════════════════════════════════════════ */

async function fetchLiveData() {
  try {
    const res = await fetch(`/stream-data?device=${currentDevice}`);
    if (!res.ok) throw new Error('fetch failed');
    const row = await res.json();
    return {
      cpu_temp: parseFloat(row.cpu_temp)           || 0,
      gpu_temp: parseFloat(row.gpu_temp)           || 0,
      cpu_use:  parseFloat(row.cpu_usage)          || 0,
      gpu_use:  parseFloat(row.gpu_usage)          || 0,
      ram:      parseFloat(row.ram_usage)          || 0,
      pwr:      parseFloat(row.power_consumption)  || 0,
      volt:     parseFloat(row.voltage)            || 0,
    };
  } catch (err) {
    return simulateMetrics();
  }
}

function simulateMetrics() {
  const t = tickCount;
  const spike = Math.random() < 0.07 ? 1.22 : 1;
  const base = currentDevice === 'rpi'
    ? { cpuT:45, gpuT:38, cpuU:40, gpuU:35, ram:52, pwr:4.2, volt:5.1 }
    : { cpuT:55, gpuT:62, cpuU:50, gpuU:58, ram:64, pwr:9.8, volt:7.6 };
  const n = (v,a) => +(v + (Math.random()-.5)*a*2).toFixed(2);
  return {
    cpu_temp: n(base.cpuT + Math.sin(t/12)*4, 2)   * spike,
    gpu_temp: n(base.gpuT + Math.sin(t/15)*5, 2.5) * spike,
    cpu_use:  Math.min(99, n(base.cpuU + Math.sin(t/8)*10,  5) * spike),
    gpu_use:  Math.min(99, n(base.gpuU + Math.sin(t/10)*12, 6) * spike),
    ram:      Math.min(99, n(base.ram  + Math.sin(t/20)*6,  2)),
    pwr:      n(base.pwr  + Math.sin(t/10)*.8, .3) * spike,
    volt:     n(base.volt - Math.sin(t/15)*.05, .04),
  };
}

/* ══════════════════════════════════════════════
   ML PREDICTION
══════════════════════════════════════════════ */

async function getMLPrediction(m) {
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({
        cpu_temp:          m.cpu_temp,
        gpu_temp:          m.gpu_temp,
        cpu_usage:         m.cpu_use,
        gpu_usage:         m.gpu_use,
        ram_usage:         m.ram,
        power_consumption: m.pwr,
        voltage:           m.volt,
        temp_diff:         m.cpu_temp - m.gpu_temp,
        load_index:        (m.cpu_use + m.gpu_use) / 2,
        device_type:       currentDevice,
      }),
    });
    if (!res.ok) throw new Error('predict failed');
    return await res.json();
  } catch (err) {
    return null;
  }
}

/* ══════════════════════════════════════════════
   RENDER — METRIC CARDS (top 4 only)
══════════════════════════════════════════════ */

function renderMetrics(m) {
  const grid = document.getElementById('metrics-grid');
  if (!grid) return;

  if (!metricsBuilt) {
    grid.innerHTML = METRIC_DEFS.map((d, i) => `
      <div class="metric-card ok" id="mc-${d.key}" style="animation-delay:${i*0.07}s">
        <div class="metric-header">
          <span class="metric-label">${d.label}</span>
          <span class="metric-dot" id="msd-${d.key}"></span>
        </div>
        <span class="metric-value" id="mv-${d.key}">--</span
        ><span class="metric-unit">${d.unit}</span>
        <div class="metric-bar"><div class="metric-bar-fill" id="mb-${d.key}" style="width:0%"></div></div>
        <div class="metric-sub">${d.sub}</div>
      </div>
    `).join('');
    metricsBuilt = true;
  }

  METRIC_DEFS.forEach(def => {
    const val    = m[def.key] || 0;
    const status = getStatus(def, val);
    const pct    = barPct(def, val);

    const card = document.getElementById(`mc-${def.key}`);
    if (card) card.className = `metric-card ${status} loaded`;

    const v = document.getElementById(`mv-${def.key}`);
    if (v) v.textContent = val.toFixed(def.unit === '%' ? 0 : 1);

    const b = document.getElementById(`mb-${def.key}`);
    if (b) b.style.width = pct.toFixed(1) + '%';
  });
}

/* ══════════════════════════════════════════════
   RENDER — HEALTH PANEL
══════════════════════════════════════════════ */

async function renderHealth(m) {
  if (!m) return;

  const panel = document.getElementById('health-panel');
  const txt   = document.getElementById('health-text');
  const sub   = document.getElementById('health-sub');
  const hmm   = document.getElementById('hmm');
  if (!panel || !txt || !sub || !hmm) return;

  const status = getOverallStatus(m);
  panel.className = `health-panel ${status}`;
  txt.textContent = status === 'ok' ? 'NORMAL' : status === 'warn' ? 'WARNING' : 'FAILURE';
  sub.textContent = status === 'ok' ? 'ALL SYSTEMS GO' : status === 'warn' ? 'ANOMALY DETECTED' : 'CRITICAL ALERT';

  const rows = [
    { label:'CPU',  val:m.cpu_use,  max:100, clr:'#00D1FF' },
    { label:'GPU',  val:m.gpu_use,  max:100, clr:'#00D1FF' },
    { label:'RAM',  val:m.ram,      max:100, clr:'#00FF9A' },
    { label:'TEMP', val:m.cpu_temp, max:90,  clr: m.cpu_temp > 75 ? '#FFB020' : '#00D1FF' },
  ];
  hmm.innerHTML = rows.map(r => {
    const pct = Math.min(100, (r.val / r.max) * 100);
    return `
      <div class="hmm-row">
        <span class="hmm-lbl">${r.label}</span>
        <div class="hmm-bar"><div class="hmm-fill" style="width:${pct.toFixed(1)}%;background:${r.clr};"></div></div>
        <span class="hmm-val">${r.val.toFixed(0)}${r.label==='TEMP'?'°':'%'}</span>
      </div>`;
  }).join('');

  /* ML */
  try {
    const ml = await getMLPrediction(m);
    if (ml && ml.prediction !== undefined) {
      const pred = String(ml.prediction).toLowerCase();
      const conf = parseFloat(ml.confidence);
      const risk = parseFloat(ml.failure_risk);
      lastMLPred        = pred;
      lastMLConfidence  = conf;
      lastMLFailureRisk = risk;

      const mlScore    = document.getElementById('ml-score');
      const riskPct    = document.getElementById('risk-pct');
      const riskBar    = document.getElementById('risk-bar');
      const predLabel  = document.getElementById('ml-pred-label');

      if (mlScore) { mlScore.textContent = conf.toFixed(1)+'%'; mlScore.className = `ml-score ${pred==='normal'?'':pred}`; }
      if (riskPct) riskPct.textContent = risk.toFixed(1)+'%';
      if (riskBar) riskBar.style.width = Math.min(100, risk).toFixed(1)+'%';
      if (predLabel) {
        const lmap = { normal:'NORMAL OPERATION', warning:'ANOMALY DETECTED', failure:'FAILURE PREDICTED' };
        predLabel.textContent = lmap[pred] || pred.toUpperCase();
        predLabel.className   = `ml-pred-label ${pred==='normal'?'ok':pred}`;
      }

      if (pred === 'failure' && tickCount % 5 === 0)
        addBotMessage(`⚠ CRITICAL: ML predicts imminent failure (confidence ${conf.toFixed(1)}%). Immediate action required.`);
      else if (pred === 'warning' && tickCount % 15 === 0)
        addBotMessage(`⚠ Warning: Anomaly patterns detected (risk ${risk.toFixed(1)}%). Recommend inspection.`);
    }
  } catch(err) { /* silent */ }
}

/* ══════════════════════════════════════════════
   RENDER — ALERTS
══════════════════════════════════════════════ */

function renderAlerts(m) {
  const list = document.getElementById('alert-list');
  if (!list) return;

  const alerts = [];
  ALL_DEFS.forEach(def => {
    const s = getStatus(def, m[def.key] || 0);
    if (s !== 'ok') alerts.push({ status:s, msg:`${def.label}: ${(m[def.key]||0).toFixed(1)}${def.unit} — ${s==='warn'?'ELEVATED':'CRITICAL'}` });
  });

  if (alerts.length === 0) {
    list.innerHTML = `<div class="alert-item ok"><div class="adot"></div>All parameters nominal</div>`;
  } else {
    list.innerHTML = alerts.map(a =>
      `<div class="alert-item ${a.status}"><div class="adot"></div>${a.msg}</div>`
    ).join('');
  }
}

/* ══════════════════════════════════════════════
   TIMESTAMP & UPTIME
══════════════════════════════════════════════ */

function updateTimestamp() {
  const n = new Date();
  const hms = [n.getUTCHours(), n.getUTCMinutes(), n.getUTCSeconds()]
    .map(v => String(v).padStart(2,'0')).join(':');
  const ts = document.getElementById('ts');
  if (ts) ts.textContent = hms + ' UTC';
  uptimeSeconds++;
  const up = document.getElementById('uptime');
  if (up) up.textContent = formatUptime(uptimeSeconds);
}

/* ══════════════════════════════════════════════
   DEVICE SELECTOR
══════════════════════════════════════════════ */

function selectDevice(d) {
  currentDevice = d;
  document.getElementById('btn-rpi').className    = 'dev-btn' + (d==='rpi'    ? ' active' : '');
  document.getElementById('btn-jetson').className = 'dev-btn' + (d==='jetson' ? ' active' : '');

  const dev = DEVICES[d];
  document.getElementById('dev-id').textContent    = dev.id;
  document.getElementById('dev-model').textContent = dev.model;
  document.getElementById('dev-os').textContent    = dev.os;

  resetCharts();
  metricsBuilt = false;

  addBotMessage(`Switched to <b>${dev.model}</b>. Streaming telemetry from <b>${dev.id}</b>. Running ${dev.os}.`);
}

/* ══════════════════════════════════════════════
   CHATBOT
══════════════════════════════════════════════ */

function addBotMessage(text, typing = false) {
  const container = document.getElementById('chat-messages');
  if (!container) return null;

  if (typing) {
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.id = 'typing-indicator';
    el.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-bubble">
        <div class="typing">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>`;
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
    return el;
  }

  const el = document.createElement('div');
  el.className = 'msg bot';
  el.innerHTML = `<div class="msg-avatar">AI</div><div class="msg-bubble">${text}</div>`;
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
  chatHistory.push({ role:'assistant', content:text });
  if (chatHistory.length > MAX_CHAT_HISTORY) chatHistory.shift();
  return el;
}

function addUserMessage(text) {
  const container = document.getElementById('chat-messages');
  if (!container) return;
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="msg-bubble">${text}</div><div class="msg-avatar">YOU</div>`;
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
  chatHistory.push({ role:'user', content:text });
  if (chatHistory.length > MAX_CHAT_HISTORY) chatHistory.shift();
}

async function getAIResponse(userMessage) {
  const telemetry = {
    cpu_temp:        metrics?.cpu_temp ?? null,
    gpu_temp:        metrics?.gpu_temp ?? null,
    cpu_use:         metrics?.cpu_use  ?? null,
    gpu_use:         metrics?.gpu_use  ?? null,
    ram:             metrics?.ram      ?? null,
    pwr:             metrics?.pwr      ?? null,
    volt:            metrics?.volt     ?? null,
    status:          getOverallStatus(metrics || {}),
    ml_pred:         lastMLPred        || 'unknown',
    ml_confidence:   lastMLConfidence  || 0,
    ml_failure_risk: lastMLFailureRisk || 0,
  };
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ message:userMessage, device:currentDevice, telemetry }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.reply || 'No response received.';
  } catch (err) {
    return `⚠ SENTINEL AI temporarily unavailable. (${err.message})`;
  }
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  if (!input) return;
  const q = input.value.trim();
  if (!q) return;

  input.value    = '';
  input.disabled = true;
  addUserMessage(q);
  const indicator = addBotMessage('', true);
  const reply = await getAIResponse(q);
  if (indicator) indicator.remove();
  addBotMessage(reply);
  input.disabled = false;
  input.focus();
}

const QUICK_QUERIES = {
  status: 'What is the current device health status?',
  why:    'Why is the device showing this health state?',
  ttf:    'How long until potential failure?',
  fix:    'What should I do to fix or prevent issues?',
};

async function quickAction(type) {
  const query = QUICK_QUERIES[type];
  if (!query) return;
  addUserMessage(query);
  const indicator = addBotMessage('', true);
  const reply = await getAIResponse(query);
  if (indicator) indicator.remove();
  addBotMessage(reply);
}

/* ══════════════════════════════════════════════
   CHARTS
══════════════════════════════════════════════ */

function initCharts() {
  const tempCtx  = document.getElementById('tempChart');
  const usageCtx = document.getElementById('usageChart');
  if (!tempCtx || !usageCtx) return;

  const chartOpts = (yMax) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: {
        beginAtZero: false,
        max: yMax,
        ticks: { color:'#7A9AB5', font:{ size:9 } },
        grid: { color:'rgba(0,209,255,0.04)' },
      },
    },
  });

  tempChart = new Chart(tempCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label:'CPU Temp', data:[], borderColor:'#00D1FF', borderWidth:2, tension:0.4, pointRadius:0, fill:false },
        { label:'GPU Temp', data:[], borderColor:'#00FF9A', borderDash:[6,4], borderWidth:1.5, tension:0.4, pointRadius:0, fill:false },
      ],
    },
    options: chartOpts(undefined),
  });

  usageChart = new Chart(usageCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label:'CPU Usage', data:[], borderColor:'#00D1FF', borderWidth:2, tension:0.4, pointRadius:0, fill:false },
        { label:'GPU Usage', data:[], borderColor:'#FFB020', borderDash:[6,4], borderWidth:1.5, tension:0.4, pointRadius:0, fill:false },
      ],
    },
    options: { ...chartOpts(100), scales: { ...chartOpts(100).scales, y: { ...chartOpts(100).scales.y, beginAtZero:true } } },
  });
}

function updateCharts(m) {
  if (!tempChart || !usageChart) return;
  const time = new Date().toLocaleTimeString();

  tempChart.data.labels.push(time);
  tempChart.data.datasets[0].data.push(m.cpu_temp);
  tempChart.data.datasets[1].data.push(m.gpu_temp);

  usageChart.data.labels.push(time);
  usageChart.data.datasets[0].data.push(m.cpu_use);
  usageChart.data.datasets[1].data.push(m.gpu_use);

  if (tempChart.data.labels.length  > MAX_HISTORY) { tempChart.data.labels.shift();  tempChart.data.datasets.forEach(d=>d.data.shift()); }
  if (usageChart.data.labels.length > MAX_HISTORY) { usageChart.data.labels.shift(); usageChart.data.datasets.forEach(d=>d.data.shift()); }

  tempChart.update();
  usageChart.update();
}

function resetCharts() {
  if (!tempChart || !usageChart) return;
  [tempChart, usageChart].forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(d => d.data = []);
    c.update();
  });
}

/* ══════════════════════════════════════════════
   MAIN TICK
══════════════════════════════════════════════ */

async function tick() {
  try {
    tickCount++;
    metrics = await fetchLiveData();
    renderMetrics(metrics);
    renderAlerts(metrics);
    await renderHealth(metrics);
    updateCharts(metrics);
    updateTimestamp();
  } catch (err) {
    console.error('Tick error:', err);
  }
}

/* ══════════════════════════════════════════════
   INIT
══════════════════════════════════════════════ */

window.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    initCharts();
    tick();
    setInterval(tick, 10000);
    addBotMessage(
      'SENTINEL AI online. Connected to <b>RPi-4B-001</b>. ' +
      'Streaming live telemetry. All systems nominal.'
    );
  }, 200);
});