/* ═══════════════════════════════════════════════════════════════
   DAEDALUS — Simulation Engine (Dashboard Edition)
   Full auction simulation with 8 adversarial agents
   ═══════════════════════════════════════════════════════════════ */

const COLORS = { truthful:'#10b981', shader:'#f59e0b', colluder:'#f43f5e', dropout:'#7c3aed', exploiter:'#3b82f6' };
const LABELS = { truthful:'Truthful', shader:'Shader', colluder:'Colluder', dropout:'Dropout', exploiter:'Exploiter' };
const MAX_ROUNDS = 50;

let S = null; // Simulation state
let autoId = null;
let charts = {};
let aiMode = false;
let designerStatus = { status: 'idle', error: null }; // populated from /api/designer/status
let designerPollId = null;

// ── Tab Switching ────────────────────────────────────
function switchTab(id) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  if (event) event.target.classList.add('active');
  document.getElementById('sim-controls').style.display = id === 'sim' ? 'flex' : 'none';
  if (id === 'sim') setTimeout(resizeCharts, 50);
}

function toggleAIMode() {
  aiMode = !aiMode;
  if (aiMode) {
    aiLog('AI mode enabled — warming up designer...', 'info');
    fetch('/api/designer/warmup', { method: 'POST' })
      .then(r => r.json())
      .then(s => {
        designerStatus = s;
        renderDesignerStatus();
        aiLog(`Warmup ack — status=${s.status || 'idle'}`, 'info');
      })
      .catch(e => aiLog(`Warmup request failed: ${e}`, 'error'));
    startDesignerPolling();
  } else {
    aiLog('AI mode disabled', 'info');
    stopDesignerPolling();
  }
  renderDesignerStatus();
}

// ── AI Log ───────────────────────────────────────────
function aiLog(msg, severity) {
  const el = document.getElementById('ai-log');
  if (!el) return;
  const empty = el.querySelector('.ai-log-empty');
  if (empty) empty.remove();
  const colors = {
    error:  'var(--rose)',
    warn:   'var(--amber)',
    ok:     'var(--emerald)',
    info:   'var(--t4)',
  };
  const col = colors[severity] || colors.info;
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const row = document.createElement('div');
  row.style.cssText = `color:${col}; margin-bottom:2px; word-break:break-word;`;
  row.textContent = `[${t}] ${msg}`;
  el.insertBefore(row, el.firstChild);
  while (el.childNodes.length > 50) el.removeChild(el.lastChild);
}

let _lastPolledStatus = null;
let _autoEngagedOnce = false;
function _logStatusTransition(prev, next, error) {
  if (prev === next) return;
  if (next === 'ready') {
    aiLog(`Designer ready — model loaded`, 'ok');
    if (!aiMode && !_autoEngagedOnce) {
      _autoEngagedOnce = true;
      aiMode = true;
      aiLog('AI mode auto-engaged — simulation will now use real adapter inferences', 'ok');
      renderDesignerStatus();
    }
  } else if (next === 'loading') {
    aiLog('Designer loading (downloading weights)…', 'info');
  } else if (next === 'error') {
    aiLog(`Designer load FAILED: ${error || 'unknown'}`, 'error');
  } else if (next === 'idle') {
    aiLog('Designer idle', 'info');
  }
}

function renderDesignerStatus() {
  const btn = document.getElementById('btn-ai-mode');
  const sub = document.getElementById('ai-status-sub');
  if (!btn) return;

  const s = designerStatus.status || 'idle';
  let dotColor = 'var(--t4)';
  let label = 'Enable AI Mode';
  let glow = '';
  let cls = '';

  if (!aiMode) {
    if (s === 'ready') {
      label = 'Engage AI →';
      dotColor = 'var(--emerald)';
      glow = ' box-shadow: 0 0 10px var(--emerald); animation: ai-pulse 1.6s infinite;';
    } else {
      label = 'Enable AI Mode';
      dotColor = s === 'loading' ? 'var(--amber)' : (s === 'error' ? 'var(--rose)' : 'var(--t4)');
    }
  } else if (s === 'ready') {
    label = 'AI Active';
    dotColor = 'var(--cyan)';
    glow = ' box-shadow: 0 0 8px var(--cyan);';
    cls = 'active-ai';
  } else if (s === 'loading') {
    label = 'AI Loading...';
    dotColor = 'var(--amber)';
    glow = ' box-shadow: 0 0 8px var(--amber);';
    cls = 'active-ai';
  } else if (s === 'error') {
    label = 'AI Error - retry';
    dotColor = 'var(--rose)';
    cls = 'active-ai';
  } else {
    label = 'AI Starting...';
    dotColor = 'var(--amber)';
    cls = 'active-ai';
  }

  btn.innerHTML = `<span id="ai-status-dot" style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${dotColor}; margin-right:8px;${glow}"></span> ${label}`;
  btn.classList.toggle('active-ai', !!cls);

  if (sub) {
    if (s === 'ready') {
      sub.textContent = `Connected: ${designerStatus.adapter || 'kabilesh-c/daedalus-designer'}`;
      sub.style.color = 'var(--emerald)';
    } else if (s === 'loading') {
      sub.textContent = 'Downloading base model + LoRA adapter...';
      sub.style.color = 'var(--amber)';
    } else if (s === 'error') {
      sub.textContent = `Load failed: ${designerStatus.error || 'unknown'}`;
      sub.style.color = 'var(--rose)';
    } else {
      sub.textContent = 'Using kabilesh-c/daedalus-designer';
      sub.style.color = 'var(--t4)';
    }
  }
}

async function pollDesignerStatus() {
  try {
    const r = await fetch('/api/designer/status');
    designerStatus = await r.json();
  } catch (e) {
    designerStatus = { status: 'error', error: 'Server unreachable' };
  }
  _logStatusTransition(_lastPolledStatus, designerStatus.status, designerStatus.error);
  _lastPolledStatus = designerStatus.status;
  renderDesignerStatus();
  if (designerStatus.status === 'ready' || designerStatus.status === 'error') {
    stopDesignerPolling();
  }
}

function startDesignerPolling() {
  if (designerPollId) return;
  pollDesignerStatus();
  designerPollId = setInterval(pollDesignerStatus, 2000);
}

function stopDesignerPolling() {
  if (designerPollId) {
    clearInterval(designerPollId);
    designerPollId = null;
  }
}

// ── Agent Factory ────────────────────────────────────
function makeAgents() {
  return [
    { id:0, n:'A1', t:'truthful',  v:0, bid:0, sur:0, on:true, bud:Infinity, shade:0, wins:0 },
    { id:1, n:'A2', t:'truthful',  v:0, bid:0, sur:0, on:true, bud:Infinity, shade:0, wins:0 },
    { id:2, n:'A3', t:'shader',    v:0, bid:0, sur:0, on:true, bud:Infinity, shade:0.15, wins:0 },
    { id:3, n:'A4', t:'shader',    v:0, bid:0, sur:0, on:true, bud:Infinity, shade:0.18, wins:0 },
    { id:4, n:'A5', t:'colluder',  v:0, bid:0, sur:0, on:true, bud:Infinity, shade:0, wins:0, partner:5, turn:false },
    { id:5, n:'A6', t:'colluder',  v:0, bid:0, sur:0, on:true, bud:Infinity, shade:0, wins:0, partner:4, turn:true },
    { id:6, n:'A7', t:'dropout',   v:0, bid:0, sur:0, on:true, bud:Infinity, shade:0, wins:0, thresh:0.08, cumSur:0 },
    { id:7, n:'A8', t:'exploiter', v:0, bid:0, sur:0, on:true, bud:2.5, shade:0, wins:0 },
  ];
}

function readMech() {
  return {
    auction: document.getElementById('ctrl-auction').value,
    revReserve: document.getElementById('t-reserve').classList.contains('active'),
    revClearing: document.getElementById('t-clearing').classList.contains('active'),
    revWinner: document.getElementById('t-winner').classList.contains('active'),
    revBids: document.getElementById('t-bids').classList.contains('active'),
    revDist: document.getElementById('t-dist').classList.contains('active'),
    reserve: parseFloat(document.getElementById('ctrl-reserve').value),
    penShill: parseFloat(document.getElementById('ctrl-shill').value),
    penWithdraw: parseFloat(document.getElementById('ctrl-withdraw').value),
    penCollusion: parseFloat(document.getElementById('ctrl-collusion').value),
    coalition: document.getElementById('ctrl-coalition').value,
  };
}

// ── Init ─────────────────────────────────────────────
function initSim() {
  S = {
    round: 0,
    agents: makeAgents(),
    mech: readMech(),
    hist: { w:[], f:[], p:[], s:[], c:[], prices:[] },
    winner: -1,
    clearPrice: 0,
    timeline: [],
    lastMech: null,
    colRot: 0,
  };
  refreshVals();
  render();
  updateMetrics();
  initCharts();
  renderTimeline();
}

function refreshVals() {
  S.agents.forEach(a => {
    if (!a.on) return;
    const ranges = { truthful:[0.3,0.6], shader:[0.4,0.5], colluder:[0.35,0.5], dropout:[0.15,0.45], exploiter:[0.2,0.4] };
    const [lo, span] = ranges[a.t];
    a.v = lo + Math.random() * span;
  });
}

// ── AI request (returns mechanism object on success, null on failure) ─
//
// IMPORTANT: this function NEVER returns synthetic / default mechanism data.
// Every failure path:
//   1. logs a clear message to the in-page AI log,
//   2. updates designerStatus + the AI button,
//   3. pauses auto-mode on hard failures so the simulation does not
//      silently run with stale or non-AI rules,
//   4. returns null so the caller can abort the round.
async function fetchAIMechanism() {
  const obs = {
    round_number: S.round,
    market_outcomes: S.hist.w.map((w, i) => ({
      welfare_ratio: w,
      gini_coefficient: 1 - S.hist.f[i],
      participation_rate: S.hist.p[i],
      composite_reward: S.hist.c[i],
    })),
  };

  let res;
  try {
    res = await fetch('/api/design', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(obs),
    });
  } catch (e) {
    designerStatus = { status: 'error', error: 'Server unreachable: ' + e };
    renderDesignerStatus();
    aiLog(`Network error contacting /api/design: ${e}`, 'error');
    pauseAuto('AI server unreachable');
    return null;
  }

  let payload = null;
  try {
    payload = await res.json();
  } catch (e) {
    aiLog(`HTTP ${res.status}: response was not valid JSON`, 'error');
    pauseAuto('Bad response from /api/design');
    return null;
  }

  if (!res.ok) {
    designerStatus = {
      status: payload.status || 'error',
      error: payload.error || payload.detail || `HTTP ${res.status}`,
      adapter: payload.adapter,
    };
    renderDesignerStatus();

    if (res.status === 503 && payload.status === 'loading') {
      aiLog(`HTTP 503: ${payload.detail || 'designer loading'} — round skipped, will retry`, 'warn');
      startDesignerPolling();
      // soft error: keep auto running so it retries when ready
      return null;
    }

    const reason = payload.error || payload.detail || `HTTP ${res.status}`;
    if (payload.raw) {
      aiLog(`HTTP ${res.status}: ${payload.detail || 'bad model output'}`, 'error');
      aiLog(`  raw → ${String(payload.raw).slice(0, 200)}`, 'error');
    } else {
      aiLog(`HTTP ${res.status}: ${reason}`, 'error');
    }
    pauseAuto(`AI error (HTTP ${res.status})`);
    return null;
  }

  const mech = payload && payload.mechanism;
  if (!mech || typeof mech !== 'object') {
    aiLog(`Server returned 200 but body had no mechanism object`, 'error');
    pauseAuto('Empty AI response');
    return null;
  }

  designerStatus = {
    status: payload.status || 'ready',
    error: payload.error || null,
    adapter: designerStatus.adapter,
  };
  renderDesignerStatus();
  return mech;
}

function pauseAuto(reason) {
  if (!autoId) return;
  clearInterval(autoId);
  autoId = null;
  const btn = document.getElementById('btn-auto');
  if (btn) {
    btn.textContent = '▶ Auto';
    btn.className = 'h-btn h-btn-secondary';
  }
  aiLog(`Auto-step paused — ${reason}`, 'warn');
}

// ── Step ─────────────────────────────────────────────
async function stepSim() {
  if (!S || S.round >= MAX_ROUNDS) return;

  let m;
  let mechSource = 'manual';

  if (aiMode) {
    const aiMech = await fetchAIMechanism();
    if (aiMech === null) {
      // AI unavailable. NO FALLBACK — abort the round entirely.
      // Errors are already pushed to the AI log by fetchAIMechanism().
      return;
    }
    m = { ...readMech(), ...mapAiMechToUiKeys(aiMech) };
    applyMechToUI(aiMech);
    mechSource = 'ai';
    aiLog(`Round ${S.round + 1}: AI mechanism received (${aiMech.auction_type || '?'}, reserve=${(aiMech.reserve_price ?? 0).toFixed(2)})`, 'ok');
  } else {
    m = readMech();
    // Loud notice: model is ready but the toggle is off, so we are running
    // on slider values, NOT on adapter inferences.
    if (designerStatus.status === 'ready') {
      aiLog(`Round ${S.round + 1}: AI mode is OFF — using slider values (click "Enable AI Mode" to call the adapter)`, 'warn');
    }
  }
  S.lastSource = mechSource;

  S.mech = m;
  S.round++;
  refreshVals();

  const active = S.agents.filter(a => a.on);
  const n = active.length;

  // Compute bids
  S.agents.forEach(a => {
    if (!a.on) { a.bid = 0; return; }
    switch (a.t) {
      case 'truthful':
        a.bid = a.v;
        break;
      case 'shader': {
        if (m.auction === 'first_price') {
          let sf = (n - 1) / Math.max(n, 2);
          if (m.revClearing) sf -= 0.03;
          if (m.revBids) sf -= 0.05;
          if (S.hist.prices.length > 2) {
            const avg = S.hist.prices.slice(-5).reduce((s,x)=>s+x,0) / Math.min(S.hist.prices.length, 5);
            if (avg < a.v * 0.7) sf -= 0.05;
          }
          a.shade = 1 - sf;
          a.bid = a.v * sf;
        } else {
          a.shade = 0.02 + Math.random() * 0.03;
          a.bid = a.v * (1 - a.shade);
        }
        break;
      }
      case 'colluder': {
        const shouldWin = (S.colRot % 2 === 0) === a.turn;
        const canCol = m.coalition === 'allow' || (m.coalition === 'restrict' && Math.random() > 0.5);
        const risk = m.penCollusion > 1 ? 0.4 : 0;
        if (canCol && Math.random() > risk) {
          if (m.revWinner) {
            a.bid = shouldWin ? m.reserve + 0.01 : m.reserve * 0.3;
          } else {
            a.bid = shouldWin ? Math.min(a.v, Math.max(m.reserve + 0.02 + Math.random()*0.05, a.v*0.5)) : m.reserve * 0.5;
          }
        } else {
          a.bid = a.v * (m.auction === 'first_price' ? 0.85 : 0.98);
        }
        break;
      }
      case 'dropout': {
        const es = a.v - m.reserve - 0.05;
        a.cumSur += es > 0 ? es * 0.1 : -0.02;
        if (a.cumSur < -a.thresh || m.reserve > a.v * 0.9) {
          a.on = false; a.bid = 0;
        } else {
          a.bid = a.v * (m.auction === 'first_price' ? 0.9 : 1.0);
        }
        break;
      }
      case 'exploiter': {
        if (a.bud <= 0) { a.on = false; a.bid = 0; }
        else {
          const agg = a.bud > 1.5 ? 0.6 : 0.3;
          a.bid = Math.min(a.v * agg, a.bud * 0.3);
        }
        break;
      }
    }
    if (a.on && a.bid < m.reserve) a.bid = 0;
  });

  // Allocation & Payment
  const valid = S.agents.filter(a => a.on && a.bid >= m.reserve);
  valid.sort((a, b) => b.bid - a.bid);
  let winner = null, payment = 0;

  if (valid.length > 0) {
    winner = valid[0];
    switch (m.auction) {
      case 'first_price': payment = winner.bid; break;
      case 'second_price': case 'vcg': payment = valid.length > 1 ? valid[1].bid : m.reserve; break;
    }
    winner.sur = winner.v - payment;
    
    // Apply Withdrawal Penalty (chance of default)
    const defaultProb = 0.05 / (1 + m.penWithdraw); // Penalty reduces default probability
    if (Math.random() < defaultProb) {
      winner.sur -= winner.v * 0.1 + m.penWithdraw * 0.5;
      aiLog(`   [!] Winner ${winner.n} defaulted! Withdrawal penalty applied.`, 'warn');
      winner.defauted = true;
    } else {
      winner.defauted = false;
    }

    winner.wins++;
    if (winner.t === 'exploiter') winner.bud -= payment;
    
    // Apply Collusion Penalty
    if (m.penCollusion > 0 && winner.t === 'colluder') {
      const p = S.agents[winner.partner];
      if (p && p.on && p.bid < m.reserve * 0.8) {
         const pAmt = payment * m.penCollusion * 0.3;
         winner.sur -= pAmt;
         aiLog(`   [!] Collusion detected! ${winner.n} penalized ${pAmt.toFixed(3)}`, 'warn');
      }
    }

    // Apply Shill Penalty (to shaders/colluders)
    if (m.penShill > 0 && (winner.t === 'shader' || winner.t === 'colluder')) {
       if (Math.random() < 0.15 * m.penShill) {
          const pAmt = payment * m.penShill * 0.2;
          winner.sur -= pAmt;
          aiLog(`   [!] Shill-bidding suspected! ${winner.n} penalized ${pAmt.toFixed(3)}`, 'warn');
       }
    }
  }

  S.winner = winner ? winner.id : -1;
  S.clearPrice = payment;
  S.colRot++;

  // Metrics
  const met = calcMetrics(winner);
  S.hist.w.push(met.w); S.hist.f.push(met.f); S.hist.p.push(met.p); S.hist.s.push(met.s);
  S.hist.c.push(met.c); S.hist.prices.push(payment);

  // Adapt agents
  adaptAgents();

  // Render
  render();
  updateMetrics();
  updateCharts();
  updateTimeline(m);
  const srcTag = S.lastSource === 'ai' ? '  [AI]' : '';
  document.getElementById('round-badge').textContent = `Round ${S.round} / ${MAX_ROUNDS}${srcTag}`;
  document.getElementById('comp-badge').textContent = `R = ${met.c.toFixed(3)}`;
  document.getElementById('comp-badge').style.color = met.c > 0.3 ? 'var(--emerald)' : met.c > 0.1 ? 'var(--amber)' : 'var(--rose)';
}

function calcMetrics(winner) {
  const all = S.agents;
  const active = all.filter(a => a.on);
  const vals = all.map(a => a.v).sort((a,b) => b-a);
  const maxW = vals[0] || 1;
  const w = winner ? Math.min(winner.v / maxW, 1) : 0;
  const surp = active.map(a => Math.max(a.sur, 0));
  const f = 1 - gini(surp);
  const p = active.length / all.length;
  let s = 1;
  if (S.hist.w.length >= 5) {
    const rec = S.hist.w.slice(-5);
    const mu = rec.reduce((a,b)=>a+b,0) / rec.length;
    const sd = Math.sqrt(rec.reduce((a,b)=>a+(b-mu)**2,0) / rec.length);
    s = Math.max(0, 1 - sd * 3);
  }
  return { w, f, p, s, c: w*f*p*s };
}

function gini(vals) {
  if (!vals.length) return 0;
  const sorted = [...vals].sort((a,b) => a-b);
  const n = sorted.length;
  const sum = sorted.reduce((a,b) => a+b, 0);
  if (sum === 0) return 0;
  let num = 0;
  for (let i = 0; i < n; i++) num += (2*(i+1)-n-1) * sorted[i];
  return num / (n * sum);
}

function adaptAgents() {
  S.agents.forEach(a => {
    if (!a.on) {
      // Dropout re-entry check
      if (a.t === 'dropout' && S.hist.p.length > 3) {
        const rp = S.hist.p.slice(-3);
        if (rp.reduce((s,x)=>s+x,0)/3 > 0.6 && S.mech.reserve < 0.3) {
          a.on = true; a.cumSur = 0;
        }
      }
      return;
    }
    if (a.t === 'shader') {
      a.shade = S.winner === a.id ? Math.min(a.shade + 0.01, 0.35) : Math.max(a.shade - 0.005, 0.02);
    }
    if (a.t === 'colluder' && S.mech.penCollusion > 1.5) {
      a.shade = Math.min(a.shade + 0.02, 0.5);
    }
  });
}

// ── Render Agents ────────────────────────────────────
function render() {
  const grid = document.getElementById('arena-grid');
  grid.innerHTML = '';
  S.agents.forEach(a => {
    const col = COLORS[a.t];
    const bidPct = Math.min(a.bid * 100, 100);
    const valPct = Math.min(a.v * 100, 100);
    const isWin = S.winner === a.id;
    const div = document.createElement('div');
    div.className = `agent-cell${isWin ? ' winner' : ''}${!a.on ? ' out' : ''}`;
    div.innerHTML = `
      <div class="agent-dot" style="background:${col}">${a.n}</div>
      <div class="agent-label">${LABELS[a.t]}</div>
      <div class="agent-status" style="color:${isWin ? 'var(--cyan)' : a.on ? 'var(--t4)' : 'var(--rose)'}">
        ${isWin ? '🏆 WIN' : a.on ? 'ACTIVE' : '❌ OUT'}
      </div>
      ${a.t === 'colluder' ? '<div class="paired-tag">🔗 paired</div>' : ''}
      <div class="bid-wrap">
        <div class="val-line" style="bottom:${valPct}%"></div>
        <div class="bid-fill" style="height:${bidPct}%;background:${col}30;border-left:2px solid ${col}"></div>
      </div>
      <div class="bid-nums">
        <span class="bn-bid" style="color:${col}">${a.on ? a.bid.toFixed(3) : '—'}</span>
        <span class="bn-val">v:${a.v.toFixed(2)}</span>
      </div>`;
    grid.appendChild(div);
  });
}

function updateMetrics() {
  const r = S.round;
  const w = r > 0 ? S.hist.w[r-1] : 0;
  const f = r > 0 ? S.hist.f[r-1] : 0;
  const p = r > 0 ? S.hist.p[r-1] : 0;
  const s = r > 0 ? S.hist.s[r-1] : 0;
  const c = r > 0 ? S.hist.c[r-1] : 0;
  document.getElementById('mv-w').textContent = w.toFixed(2);
  document.getElementById('mv-f').textContent = f.toFixed(2);
  document.getElementById('mv-p').textContent = p.toFixed(2);
  document.getElementById('mv-s').textContent = s.toFixed(2);
  document.getElementById('mb-w').style.width = `${w*100}%`;
  document.getElementById('mb-f').style.width = `${f*100}%`;
  document.getElementById('mb-p').style.width = `${p*100}%`;
  document.getElementById('mb-s').style.width = `${s*100}%`;
  document.getElementById('comp-val').textContent = c.toFixed(3);
}

// ── Charts ───────────────────────────────────────────
function initCharts() {
  ['w','f','p'].forEach(k => {
    const canvas = document.getElementById('ch-' + k);
    charts[k] = { canvas, ctx: canvas.getContext('2d'), key: k };
  });
  resizeCharts();
}

function resizeCharts() {
  Object.values(charts).forEach(ch => {
    if (!ch.canvas) return;
    const rect = ch.canvas.parentElement.getBoundingClientRect();
    ch.canvas.width = Math.max(rect.width, 100);
    ch.canvas.height = Math.max(rect.height - 16, 30);
    drawChart(ch);
  });
}

function drawChart(ch) {
  const { canvas: cv, ctx, key } = ch;
  const colors = { w:'#10b981', f:'#7c3aed', p:'#f59e0b' };
  const col = colors[key];
  const data = S.hist[key];
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.03)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { ctx.beginPath(); ctx.moveTo(0, H/4*i); ctx.lineTo(W, H/4*i); ctx.stroke(); }
  if (data.length < 2) return;
  const sx = W / Math.max(MAX_ROUNDS - 1, 1);
  // Area
  ctx.beginPath(); ctx.moveTo(0, H);
  data.forEach((v, i) => ctx.lineTo(i * sx, H - v * H));
  ctx.lineTo((data.length-1)*sx, H); ctx.closePath();
  ctx.fillStyle = col + '12'; ctx.fill();
  // Line
  ctx.beginPath();
  data.forEach((v, i) => { if (i===0) ctx.moveTo(0, H-v*H); else ctx.lineTo(i*sx, H-v*H); });
  ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
  // Dot
  const last = data[data.length-1];
  ctx.beginPath(); ctx.arc((data.length-1)*sx, H-last*H, 3, 0, Math.PI*2);
  ctx.fillStyle = col; ctx.fill();
}

function updateCharts() { Object.values(charts).forEach(ch => ch && drawChart(ch)); }

// ── Timeline ─────────────────────────────────────────
function updateTimeline(m) {
  const prev = S.lastMech;
  let desc = null;
  if (!prev) { desc = `Init: ${m.auction.replace('_','-')}`; }
  else {
    const ch = [];
    if (prev.auction !== m.auction) ch.push(`Auction→${m.auction.replace('_','-')}`);
    if (Math.abs(prev.reserve - m.reserve) > 0.02) ch.push(`Reserve${m.reserve>prev.reserve?'↑':'↓'}${m.reserve.toFixed(2)}`);
    if (prev.revWinner !== m.revWinner) ch.push(`Winner ${m.revWinner?'shown':'hidden'}`);
    if (prev.penCollusion !== m.penCollusion) ch.push(`ColPen→${m.penCollusion.toFixed(1)}`);
    if (prev.coalition !== m.coalition) ch.push(`Coalition→${m.coalition}`);
    if (ch.length) desc = ch.join(', ');
  }
  S.lastMech = { ...m };
  if (desc) {
    const c = S.hist.c;
    const cur = c[c.length-1] || 0;
    const pre = c.length > 1 ? c[c.length-2] : cur;
    S.timeline.push({ round:S.round, action:desc, delta:cur-pre, comp:cur });
    renderTimeline();
  }
}

function renderTimeline() {
  const el = document.getElementById('timeline');
  el.innerHTML = '';
  S.timeline.slice(-12).forEach(e => {
    const cls = e.delta > 0.01 ? 'pos' : e.delta < -0.01 ? 'neg' : '';
    el.innerHTML += `<div class="t-event"><div class="t-round">R${e.round}</div><div class="t-action">${e.action}</div><div class="t-impact ${cls}">${e.delta>=0?'+':''}${e.delta.toFixed(3)} → ${e.comp.toFixed(3)}</div></div>`;
  });
  el.scrollLeft = el.scrollWidth;
}

// ── Controls ─────────────────────────────────────────
function togChip(el) { el.classList.toggle('active'); }
function onCtrlChange() { /* applied on next step */ }

function mapAiMechToUiKeys(m) {
  if (!m) return {};
  const out = {};
  if (m.auction_type) out.auction = m.auction_type;
  if (m.reserve_price !== undefined) out.reserve = m.reserve_price;
  if (m.reveal_reserve !== undefined) out.revReserve = !!m.reveal_reserve;
  if (m.reveal_clearing_price !== undefined) out.revClearing = !!m.reveal_clearing_price;
  if (m.reveal_winner_identity !== undefined) out.revWinner = !!m.reveal_winner_identity;
  if (m.reveal_competing_bids !== undefined) out.revBids = !!m.reveal_competing_bids;
  if (m.reveal_bid_distribution !== undefined) out.revDist = !!m.reveal_bid_distribution;
  if (m.shill_penalty !== undefined) out.penShill = m.shill_penalty;
  if (m.withdrawal_penalty !== undefined) out.penWithdraw = m.withdrawal_penalty;
  if (m.collusion_penalty !== undefined) out.penCollusion = m.collusion_penalty;
  if (m.coalition_policy) out.coalition = m.coalition_policy;
  return out;
}

function applyMechToUI(m) {
  if (!m) return;
  if (m.auction_type) document.getElementById('ctrl-auction').value = m.auction_type;
  if (m.reserve_price !== undefined) {
    document.getElementById('ctrl-reserve').value = m.reserve_price;
    document.getElementById('sv-reserve').textContent = m.reserve_price.toFixed(2);
  }
  if (m.shill_penalty !== undefined) {
    document.getElementById('ctrl-shill').value = m.shill_penalty;
    document.getElementById('sv-shill').textContent = m.shill_penalty.toFixed(1);
  }
  if (m.withdrawal_penalty !== undefined) {
    document.getElementById('ctrl-withdraw').value = m.withdrawal_penalty;
    document.getElementById('sv-withdraw').textContent = m.withdrawal_penalty.toFixed(1);
  }
  if (m.collusion_penalty !== undefined) {
    document.getElementById('ctrl-collusion').value = m.collusion_penalty;
    document.getElementById('sv-collusion').textContent = m.collusion_penalty.toFixed(1);
  }
  if (m.coalition_policy) document.getElementById('ctrl-coalition').value = m.coalition_policy;
  
  // Set toggle chips
  const chips = { 
    't-reserve': m.reveal_reserve, 
    't-clearing': m.reveal_clearing_price, 
    't-winner': m.reveal_winner_identity, 
    't-bids': m.reveal_competing_bids, 
    't-dist': m.reveal_bid_distribution 
  };
  for (let id in chips) {
    const el = document.getElementById(id);
    if (!el) continue;
    if (chips[id]) el.classList.add('active'); else el.classList.remove('active');
  }
}

function toggleAuto() {
  const btn = document.getElementById('btn-auto');
  if (autoId) {
    clearInterval(autoId); autoId = null;
    btn.textContent = '▶ Auto'; btn.className = 'h-btn h-btn-secondary';
  } else {
    autoId = setInterval(async () => {
      if (S.round >= MAX_ROUNDS) { clearInterval(autoId); autoId = null; btn.textContent = '▶ Auto'; btn.className = 'h-btn h-btn-secondary'; return; }
      await stepSim();
    }, 400);
    btn.textContent = '⏸ Pause'; btn.className = 'h-btn h-btn-primary';
  }
}

function resetSim() {
  if (autoId) {
    clearInterval(autoId);
    autoId = null;
    document.getElementById('btn-auto').textContent = '▶ Auto';
    document.getElementById('btn-auto').className = 'h-btn h-btn-secondary';
  }
  document.getElementById('round-badge').textContent = 'Round 0 / 50';
  document.getElementById('comp-badge').textContent = 'R = 0.000';
  document.getElementById('comp-badge').style.color = 'var(--cyan)';

  // Clear the AI log so stale "AI mode is OFF" / error entries from the
  // previous run don't bleed into the next session.
  const logEl = document.getElementById('ai-log');
  if (logEl) {
    logEl.innerHTML = '<div class="ai-log-empty" style="opacity:0.5;">AI log will appear here…</div>';
  }

  // Re-arm auto-engage so a freshly-reset run with a ready designer ends up
  // calling the adapter, instead of silently inheriting an "off" toggle from
  // the previous session.
  _autoEngagedOnce = false;
  if (designerStatus && designerStatus.status === 'ready' && !aiMode) {
    aiMode = true;
    _autoEngagedOnce = true;
    aiLog('Reset — AI mode auto-engaged (designer is ready)', 'ok');
    renderDesignerStatus();
  } else if (aiMode) {
    aiLog('Reset — AI mode still active', 'info');
  } else {
    aiLog('Reset — AI mode off (designer not ready yet)', 'info');
  }

  initSim();
}

// ── Init on Load ─────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Inject the AI-button pulse keyframe so it lives without touching styles.css.
  const styleEl = document.createElement('style');
  styleEl.textContent = `
    @keyframes ai-pulse {
      0%   { box-shadow: 0 0 6px rgba(16,185,129,0.6); }
      50%  { box-shadow: 0 0 14px rgba(16,185,129,1.0); }
      100% { box-shadow: 0 0 6px rgba(16,185,129,0.6); }
    }
  `;
  document.head.appendChild(styleEl);

  initSim();
  window.addEventListener('resize', () => setTimeout(resizeCharts, 100));
  // Show server-side designer status from the start so the user sees
  // whether the model is downloading / ready / errored.
  pollDesignerStatus();
  startDesignerPolling();
});
