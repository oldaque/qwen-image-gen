/* ==========================================================================
 * Gerador de Imagens — SPA sem framework (contrato v1)
 *
 * Endpoints usados:
 *   GET    /api/v1/config
 *   GET    /api/v1/health
 *   POST   /api/v1/jobs
 *   GET    /api/v1/jobs?limit&offset
 *   GET    /api/v1/jobs/{job_id}
 *   GET    /api/v1/jobs/{job_id}/events   (SSE, evento "status")
 *   DELETE /api/v1/jobs/{job_id}
 *   GET    /api/v1/images/{image_id}
 *
 * Acompanhamento: SSE primeiro; se o SSE falhar (ou ficar mudo por 12 s)
 * cai automaticamente para polling de 1,5 s. O SQLite do servidor é a
 * fonte de verdade do status, então recarregar a página sempre funciona.
 * ========================================================================== */

'use strict';

const API_BASE = '/api/v1';
const POLL_INTERVAL_MS = 1500;
const SSE_IDLE_TIMEOUT_MS = 12000;
const HISTORY_LIMIT = 50;
const HEALTH_INTERVAL_MS = 15000;
const TOAST_TTL_MS = 5000;
const PROMPT_MAX = 4000;
const NUM_IMAGES_MIN = 1;
const NUM_IMAGES_MAX = 4;

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled']);

const STATUS_META = {
  queued: { label: 'na fila', dot: 'bg-amber-400', pulse: false },
  running: { label: 'gerando', dot: 'bg-sky-400', pulse: true },
  succeeded: { label: 'concluído', dot: 'bg-emerald-400', pulse: false },
  failed: { label: 'falhou', dot: 'bg-rose-400', pulse: false },
  cancelled: { label: 'cancelado', dot: 'bg-zinc-400', pulse: false },
};

const state = {
  config: null,
  health: null,
  jobs: [],              // JobView[]
  currentJobId: null,
  watchers: new Map(),   // jobId -> { es, pollTimer, watchdog, source, gotStatus, errors }
  lightbox: null,        // { jobId, imageId }
  submitting: false,
  onlyCurrent: false,
  notified: new Set(),   // jobs cujo status terminal já foi anunciado
};

/* --------------------------------------------------------------- helpers */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

function statusMeta(status) {
  return STATUS_META[status] || { label: status || 'desconhecido', dot: 'bg-zinc-500', pulse: false };
}

function parseTs(ts) {
  if (!ts) return null;
  let value = String(ts);
  // timestamps sem timezone são tratados como UTC (o servidor grava ISO UTC)
  if (!/[zZ]|[+-]\d\d:?\d\d$/.test(value)) value = value.replace(' ', 'T') + 'Z';
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? null : ms;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—';
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total} s`;
  const m = Math.floor(total / 60);
  const rs = total % 60;
  if (m < 60) return rs ? `${m} min ${rs} s` : `${m} min`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h} h ${rm} min` : `${h} h`;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = Number(bytes);
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  const rendered = Number.isInteger(value) ? String(value) : value >= 10 ? String(Math.round(value)) : value.toFixed(1);
  return `${rendered} ${units[i]}`;
}

function shortId(id) {
  return id ? String(id).slice(0, 8) : '—';
}

function jobElapsedSeconds(job) {
  if (typeof job.elapsed_seconds === 'number') return job.elapsed_seconds;
  const start = parseTs(job.started_at);
  if (start) return (Date.now() - start) / 1000;
  return null;
}

function jobImages(job) {
  if (!job || !Array.isArray(job.images)) return [];
  return job.images.map((img) => Object.assign({}, img, { job_id: job.job_id }));
}

function imageUrl(image) {
  return image && image.url ? image.url : `${API_BASE}/images/${encodeURIComponent(image.image_id)}`;
}

function isTerminal(job) {
  return !!job && TERMINAL.has(job.status);
}

/* ----------------------------------------------------------------- API */

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

function pickMessage(payload, status) {
  const detail = payload && typeof payload === 'object' ? payload.detail || payload.error : payload;
  if (typeof detail === 'string' && detail.trim()) return detail.trim();
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null;
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .join('; ');
  }
  if (detail && typeof detail === 'object' && typeof detail.message === 'string') return detail.message;
  return `erro ${status} do servidor`;
}

async function apiFetch(path, options = {}) {
  const opts = Object.assign({}, options);
  opts.headers = Object.assign({ Accept: 'application/json' }, options.headers || {});

  let res;
  try {
    res = await fetch(path, opts);
  } catch (cause) {
    throw new ApiError('falha de rede: não foi possível falar com o servidor', 0, null);
  }

  if (res.status === 204) return null;

  const text = await res.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (err) {
      payload = text;
    }
  }

  if (!res.ok) {
    let message = pickMessage(payload, res.status);
    if (res.status === 400 && /provider/i.test(message)) {
      message = `provider indisponível: ${message}`;
    }
    throw new ApiError(message, res.status, payload);
  }
  return payload;
}

/* --------------------------------------------------------------- toasts */

function toast(message, kind = 'info', ttl = TOAST_TTL_MS) {
  const box = $('#toasts');
  if (!box) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.dataset.kind = kind;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  el.textContent = message;
  box.appendChild(el);
  const remove = () => {
    el.classList.add('leaving');
    setTimeout(() => el.remove(), 200);
  };
  const timer = setTimeout(remove, ttl);
  el.addEventListener('click', () => {
    clearTimeout(timer);
    remove();
  });
}

/* ------------------------------------------------------- estado dos jobs */

function upsertJob(job) {
  if (!job || !job.job_id) return null;
  const idx = state.jobs.findIndex((item) => item.job_id === job.job_id);
  if (idx >= 0) {
    state.jobs[idx] = job;
  } else {
    state.jobs.unshift(job);
  }
  return job;
}

function removeJob(jobId) {
  state.jobs = state.jobs.filter((job) => job.job_id !== jobId);
  state.notified.delete(jobId);
  if (state.currentJobId === jobId) state.currentJobId = null;
  if (state.lightbox && state.lightbox.jobId === jobId) closeLightbox();
}

function findJob(jobId) {
  return state.jobs.find((job) => job.job_id === jobId) || null;
}

function sortedJobs() {
  return state.jobs.slice().sort((a, b) => {
    const ta = parseTs(a.created_at) || 0;
    const tb = parseTs(b.created_at) || 0;
    return tb - ta;
  });
}

/* ------------------------------------------------------ acompanhamento */

function stopWatcher(jobId) {
  const watcher = state.watchers.get(jobId);
  if (!watcher) return;
  if (watcher.es) {
    watcher.es.onerror = null;
    watcher.es.close();
  }
  if (watcher.pollTimer) clearTimeout(watcher.pollTimer);
  if (watcher.watchdog) clearTimeout(watcher.watchdog);
  state.watchers.delete(jobId);
}

function stopAllWatchers() {
  Array.from(state.watchers.keys()).forEach(stopWatcher);
}

function armWatchdog(jobId, watcher) {
  if (watcher.watchdog) clearTimeout(watcher.watchdog);
  watcher.watchdog = setTimeout(() => {
    fallbackToPolling(jobId, watcher, 'sem eventos do servidor');
  }, SSE_IDLE_TIMEOUT_MS);
}

function handleJobUpdate(job) {
  if (!job || !job.job_id) return;
  upsertJob(job);
  renderCurrentJob();
  renderGallery();
  renderHistory();
  if (isTerminal(job)) {
    stopWatcher(job.job_id);
    if (state.notified.has(job.job_id)) return;
    state.notified.add(job.job_id);
    const meta = statusMeta(job.status);
    if (job.status === 'succeeded') {
      toast(`job ${shortId(job.job_id)} ${meta.label} em ${formatDuration(jobElapsedSeconds(job))}`, 'ok');
    } else if (job.status === 'failed') {
      toast(`job ${shortId(job.job_id)} falhou: ${job.error || 'erro desconhecido'}`, 'error', 9000);
    } else {
      toast(`job ${shortId(job.job_id)} ${meta.label}`, 'warn');
    }
  }
}

function watchJob(jobId) {
  const job = findJob(jobId);
  if (!job) return;
  if (isTerminal(job)) {
    stopWatcher(jobId);
    return;
  }
  if (state.watchers.has(jobId)) return;

  const watcher = { es: null, pollTimer: null, watchdog: null, source: 'starting', gotStatus: false, errors: 0 };
  state.watchers.set(jobId, watcher);

  if (typeof window.EventSource !== 'function') {
    fallbackToPolling(jobId, watcher, 'navegador sem EventSource');
    return;
  }

  let es;
  try {
    es = new window.EventSource(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/events`);
  } catch (err) {
    fallbackToPolling(jobId, watcher, 'EventSource indisponível');
    return;
  }

  watcher.es = es;
  watcher.source = 'sse';
  renderCurrentJob();

  const onEvent = (event) => {
    let payload = null;
    try {
      payload = JSON.parse(event.data);
    } catch (err) {
      return;
    }
    watcher.gotStatus = true;
    watcher.errors = 0;
    watcher.source = 'sse';
    armWatchdog(jobId, watcher);
    handleJobUpdate(payload);
  };

  es.addEventListener('status', onEvent);
  es.addEventListener('message', onEvent); // defensivo: evento sem nome
  es.addEventListener('open', () => {
    armWatchdog(jobId, watcher);
    renderCurrentJob();
  });

  es.onerror = () => {
    if (watcher.gotStatus) {
      // o EventSource nativo já tenta reconectar; o watchdog cobre a morte silenciosa
      armWatchdog(jobId, watcher);
      return;
    }
    fallbackToPolling(jobId, watcher, 'conexão SSE recusada');
  };

  armWatchdog(jobId, watcher);
}

function fallbackToPolling(jobId, watcher, reason) {
  if (!state.watchers.has(jobId)) return;
  if (watcher.source === 'polling') return;
  if (watcher.es) {
    watcher.es.onerror = null;
    watcher.es.close();
    watcher.es = null;
  }
  if (watcher.watchdog) clearTimeout(watcher.watchdog);
  watcher.source = 'polling';
  renderCurrentJob();
  if (reason) toast(`acompanhamento via SSE falhou (${reason}); usando polling de 1,5 s`, 'warn', 7000);
  pollJob(jobId, watcher);
}

async function pollJob(jobId, watcher) {
  if (!state.watchers.has(jobId)) return;
  try {
    const job = await apiFetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`);
    if (!job) {
      // 404: job removido no servidor
      stopWatcher(jobId);
      removeJob(jobId);
      renderAll();
      return;
    }
    watcher.errors = 0;
    handleJobUpdate(job);
    if (isTerminal(job)) return;
  } catch (err) {
    if (err.status === 404) {
      stopWatcher(jobId);
      removeJob(jobId);
      renderAll();
      return;
    }
    watcher.errors += 1;
    if (watcher.errors === 3) {
      toast(`falha ao consultar o job: ${err.message}`, 'error', 7000);
    }
  }
  if (state.watchers.has(jobId)) {
    watcher.pollTimer = setTimeout(() => pollJob(jobId, watcher), POLL_INTERVAL_MS);
  }
}

function fetchJobOnce(jobId) {
  return apiFetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`)
    .then((job) => {
      if (job) {
        handleJobUpdate(job);
        if (!isTerminal(job)) watchJob(jobId);
      }
    })
    .catch((err) => {
      if (err.status === 404) {
        removeJob(jobId);
        renderAll();
      } else {
        toast(err.message, 'error', 7000);
      }
    });
}

/* ------------------------------------------------------------- config */

async function loadConfig() {
  try {
    const config = await apiFetch(`${API_BASE}/config`);
    state.config = config || null;
    applyConfigToForm();
    renderHeader();
    $('#configError').hidden = true;
    const warn = $('#providerWarning');
    warn.hidden = !(config && config.provider === 'local_comfy');
  } catch (err) {
    const box = $('#configError');
    box.hidden = false;
    box.textContent = `não foi possível carregar /api/v1/config — ${err.message}`;
    renderHeader();
  }
}

async function loadHealth() {
  try {
    const health = await apiFetch(`${API_BASE}/health`);
    state.health = health || null;
    renderHeader();
  } catch (err) {
    state.health = null;
    const dot = $('#healthDot');
    dot.className = 'status-dot bg-rose-500 pulse';
    $('#healthLabel').textContent = 'servidor indisponível';
  }
}

function renderHeader() {
  const config = state.config || {};
  const health = state.health || {};
  const provider = health.provider || config.provider || '—';
  const model = health.model_id || config.model_id || '—';

  $('#providerBadge').textContent = provider;
  $('#modelBadge').textContent = model;
  $('#modelBadge').title = model;

  const queue = typeof health.queue_depth === 'number' ? health.queue_depth : null;
  $('#queueBadge').textContent = queue === null ? '—' : String(queue);
  $('#queueBadge').title = queue === null ? 'fila indisponível' : `${queue} job(s) na fila`;

  $('#versionBadge').textContent = `v${health.version || config.version || '—'}`;

  if (state.health) {
    const dot = $('#healthDot');
    dot.className = 'status-dot bg-emerald-400';
    const pending = typeof queue === 'number' ? queue : 0;
    $('#healthLabel').textContent = pending > 0 ? `servidor ok · ${pending} na fila` : 'servidor ok';
  }

  const available = Array.isArray(config.providers_available) ? config.providers_available : [];
  if (available.length) {
    $('#providerBadge').parentElement.title = `disponíveis: ${available.join(', ')}`;
  }
}

async function loadJobs() {
  try {
    const data = await apiFetch(`${API_BASE}/jobs?limit=${HISTORY_LIMIT}&offset=0`);
    const items = data && Array.isArray(data.items) ? data.items : [];
    state.jobs = items;
    if (state.currentJobId && !findJob(state.currentJobId)) state.currentJobId = null;
    if (!state.currentJobId) {
      const active = sortedJobs().find((job) => !isTerminal(job));
      state.currentJobId = active ? active.job_id : null;
    }
    renderAll();
    state.jobs.filter((job) => !isTerminal(job)).forEach((job) => watchJob(job.job_id));
  } catch (err) {
    toast(`não foi possível carregar o histórico: ${err.message}`, 'error', 8000);
  }
}

async function refreshJobs() {
  await loadJobs();
  toast('histórico atualizado', 'ok', 2200);
}

/* ---------------------------------------------------------------表单 */

function applyConfigToForm() {
  const config = state.config;
  if (!config) return;
  const min = Number(config.min_size) || 512;
  const max = Number(config.max_size) || 2048;

  ['#width', '#height'].forEach((sel) => {
    const input = $(sel);
    input.min = String(min);
    input.max = String(max);
  });

  const width = clampSize(Number($('#width').value), min, max);
  const height = clampSize(Number($('#height').value), min, max);
  $('#width').value = String(width);
  $('#height').value = String(height);

  $('#sizeRangeHint').textContent =
    `entre ${min} e ${max} px, múltiplos de 8. ` +
    `timeout do job: ${formatDuration(config.job_timeout_seconds)}.`;
  syncPresetButtons();
  renderSizeHint();
}

function clampSize(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.round(value / 8) * 8));
}

function currentSizeBounds() {
  const config = state.config || {};
  return {
    min: Number(config.min_size) || 512,
    max: Number(config.max_size) || 2048,
  };
}

function syncPresetButtons() {
  const w = Number($('#width').value);
  const h = Number($('#height').value);
  $$('#presetRow .preset').forEach((btn) => {
    const size = Number(btn.dataset.preset);
    btn.classList.toggle('active', size === w && size === h);
  });
}

function renderSizeHint() {
  const w = Number($('#width').value) || 0;
  const h = Number($('#height').value) || 0;
  const n = Number($('#numImages').value) || 1;
  const mp = (w * h) / 1e6;
  let text = `${mp.toFixed(2)} MP por imagem`;
  if (state.config && state.config.provider === 'fal' && mp > 0) {
    // fal cobra por megapixel: ~$0.02/MP de saída (t2i) e ~$0.0367/MP de
    // entrada + ~$0.0367/MP de saída (edição). Estimativa grosseira.
    const refs = parseReferenceImages();
    const perImage = refs.length ? 0.036667 * 2 : 0.02;
    text += ` · estimativa fal: ~$${(mp * perImage * n).toFixed(3)}`;
  }
  $('#mpHint').textContent = text;
}

function toggleCollapsible(buttonSel, wrapSel) {
  const btn = $(buttonSel);
  const wrap = $(wrapSel);
  const open = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', open ? 'false' : 'true');
  wrap.hidden = open;
}

const FIELD_IDS = {
  prompt: 'prompt',
  negative_prompt: 'negativePrompt',
  width: 'width',
  height: 'height',
  num_images: 'numImages',
  seed: 'seed',
  reference_images: 'referenceImages',
};

function setFieldError(name, message) {
  const node = $(`[data-error-for="${name}"]`);
  if (node) node.textContent = message || '';
  const field = FIELD_IDS[name] ? $(`#${FIELD_IDS[name]}`) : null;
  if (field && field.classList) field.classList.toggle('invalid', !!message);
}

function clearFieldErrors() {
  $$('[data-error-for]').forEach((node) => {
    node.textContent = '';
  });
  $$('.input.invalid').forEach((input) => input.classList.remove('invalid'));
  const formError = $('#formError');
  formError.hidden = true;
  formError.textContent = '';
}

function showFormError(message) {
  const formError = $('#formError');
  formError.hidden = false;
  formError.textContent = message;
}

function parseReferenceImages() {
  const raw = $('#referenceImages').value || '';
  return raw
    .split(/[\n,]+/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function buildPayload() {
  const { min, max } = currentSizeBounds();
  const prompt = ($('#prompt').value || '').trim();
  const width = Number($('#width').value);
  const height = Number($('#height').value);
  const numImages = Number($('#numImages').value);
  const seedRaw = ($('#seed').value || '').trim();
  const references = parseReferenceImages();

  let ok = true;

  if (prompt.length < 1) {
    setFieldError('prompt', 'o prompt é obrigatório');
    ok = false;
  } else if (prompt.length > PROMPT_MAX) {
    setFieldError('prompt', `o prompt não pode passar de ${PROMPT_MAX} caracteres`);
    ok = false;
  }

  if (!Number.isFinite(width) || width < min || width > max) {
    setFieldError('width', `largura entre ${min} e ${max} px`);
    ok = false;
  }
  if (!Number.isFinite(height) || height < min || height > max) {
    setFieldError('height', `altura entre ${min} e ${max} px`);
    ok = false;
  }
  if (!Number.isFinite(numImages) || numImages < NUM_IMAGES_MIN || numImages > NUM_IMAGES_MAX) {
    setFieldError('num_images', `quantidade entre ${NUM_IMAGES_MIN} e ${NUM_IMAGES_MAX}`);
    ok = false;
  }
  if (seedRaw && (!/^\d+$/.test(seedRaw) || !Number.isFinite(Number(seedRaw)))) {
    setFieldError('seed', 'a seed deve ser um inteiro maior ou igual a 0');
    ok = false;
  }
  references.forEach((ref) => {
    if (!/^https?:\/\//i.test(ref) && !/^data:image\//i.test(ref)) {
      setFieldError('reference_images', 'cada referência deve ser uma URL http(s) ou um data URI de imagem');
      ok = false;
    }
  });

  if (!ok) throw new ApiError('revise os campos destacados', 0, null);

  const payload = {
    prompt,
    width: Math.round(width),
    height: Math.round(height),
    num_images: numImages,
    output_format: $('#outputFormat').value,
    prompt_expander: $('#promptExpander').value,
    reference_images: references,
  };

  const negative = ($('#negativePrompt').value || '').trim();
  if (negative) payload.negative_prompt = negative;
  if (seedRaw) payload.seed = Number(seedRaw);

  return payload;
}

function setSubmitting(active) {
  state.submitting = active;
  const btn = $('#generateBtn');
  btn.disabled = active;
  $('#generateSpinner').hidden = !active;
  $('#generateLabel').textContent = active ? 'Enviando…' : 'Gerar';
}

async function onSubmit(event) {
  event.preventDefault();
  if (state.submitting) return;
  clearFieldErrors();

  let payload;
  try {
    payload = buildPayload();
  } catch (err) {
    showFormError(err.message);
    toast(err.message, 'error', 7000);
    return;
  }

  setSubmitting(true);
  try {
    const job = await apiFetch(`${API_BASE}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    upsertJob(job);
    state.currentJobId = job.job_id;
    renderAll();
    watchJob(job.job_id);
    toast(`job ${shortId(job.job_id)} na fila`, 'ok');
    const card = $('#jobCard');
    if (card && card.scrollIntoView) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    showFormError(err.message);
    toast(err.message, 'error', 9000);
  } finally {
    setSubmitting(false);
  }
}

/* --------------------------------------------------------- cancelamentos */

async function cancelJob(jobId) {
  const job = findJob(jobId);
  if (!job) return;
  if (!window.confirm(`Cancelar o job ${shortId(jobId)}? As imagens já geradas serão apagadas.`)) return;
  try {
    await apiFetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
    stopWatcher(jobId);
    await fetchJobOnce(jobId);
    toast(`job ${shortId(jobId)} cancelado`, 'warn');
  } catch (err) {
    toast(`não foi possível cancelar: ${err.message}`, 'error', 8000);
  }
}

/** DELETE /api/v1/jobs/{job_id}: cancela (se ativo) e apaga as imagens do job. */
async function removeJobOnServer(jobId) {
  try {
    await apiFetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
    stopWatcher(jobId);
    removeJob(jobId);
    renderAll();
    toast(`job ${shortId(jobId)} removido`, 'ok');
    return true;
  } catch (err) {
    toast(`não foi possível apagar: ${err.message}`, 'error', 8000);
    return false;
  }
}

async function deleteJob(jobId) {
  const job = findJob(jobId);
  if (!job) return;
  const running = job.status === 'queued' || job.status === 'running';
  const question = running
    ? `O job ${shortId(jobId)} ainda está ${statusMeta(job.status).label}. Cancelar e apagar tudo?`
    : `Apagar o job ${shortId(jobId)} e as imagens dele?`;
  if (!window.confirm(question)) return;
  await removeJobOnServer(jobId);
}

/**
 * Apagar uma imagem.
 * O contrato expõe DELETE apenas para /api/v1/jobs/{job_id}, que apaga TODAS
 * as imagens do job. Não existe DELETE /api/v1/images/{image_id} no backend,
 * então a ação é o DELETE do job — com UMA confirmação honesta, dizendo
 * quantas imagens serão perdidas quando houver mais de uma.
 */
async function deleteImage(jobId) {
  const job = findJob(jobId);
  const total = job ? jobImages(job).length : 0;

  const question = total <= 1
    ? `Apagar a última imagem do job ${shortId(jobId)}? O job será removido junto.`
    : `O contrato não expõe DELETE por imagem. Apagar o job ${shortId(jobId)} e TODAS as ${total} imagens dele?`;
  if (!window.confirm(question)) return;

  await removeJobOnServer(jobId);
}

/* ------------------------------------------------------------ rendering */

function renderAll() {
  renderCurrentJob();
  renderGallery();
  renderHistory();
}

function renderCurrentJob() {
  const card = $('#jobCard');
  const empty = $('#jobEmpty');
  const body = $('#jobBody');
  const job = state.currentJobId ? findJob(state.currentJobId) : null;

  if (!job) {
    empty.hidden = false;
    body.hidden = true;
    body.innerHTML = '';
    return;
  }

  empty.hidden = true;
  body.hidden = false;

  const meta = statusMeta(job.status);
  const request = job.request || {};
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  const barClass =
    job.status === 'failed' ? 'is-failed' : job.status === 'cancelled' ? 'is-cancelled' : job.status === 'succeeded' ? 'is-done' : '';
  const indeterminate = job.status === 'running' && progress === 0;
  const watcher = state.watchers.get(job.job_id);
  const source =
    isTerminal(job)
      ? 'finalizado'
      : !watcher
        ? 'aguardando'
        : watcher.source === 'sse'
          ? 'ao vivo (SSE)'
          : watcher.source === 'polling'
            ? 'polling 1,5 s'
            : 'conectando…';
  const elapsed = formatDuration(jobElapsedSeconds(job));
  const images = jobImages(job);
  const refs = Array.isArray(request.reference_images) ? request.reference_images : [];
  const running = job.status === 'queued' || job.status === 'running';

  body.innerHTML = `
    <div class="flex flex-wrap items-start gap-3">
      <div class="min-w-0">
        <div class="text-[10px] uppercase tracking-widest text-zinc-500">job atual</div>
        <div class="font-mono text-sm text-zinc-300" title="${esc(job.job_id)}">#${esc(shortId(job.job_id))}</div>
      </div>
      <span class="status-chip" data-status="${esc(job.status)}">
        <span class="status-dot ${meta.dot} ${meta.pulse ? 'pulse' : ''}"></span>${esc(meta.label)}
      </span>
      <span class="badge border-ink-700 bg-ink-900 text-zinc-500" title="fonte do progresso">${esc(source)}</span>
      <div class="ml-auto flex items-center gap-2">
        ${running ? '<button type="button" class="btn-danger" data-action="cancel-job">cancelar</button>' : ''}
        <button type="button" class="btn-ghost" data-action="delete-job">apagar job</button>
      </div>
    </div>

    <div class="mt-4">
      <div class="mb-1.5 flex items-baseline justify-between text-[11px] text-zinc-500">
        <span id="jobProgressLabel">progresso</span>
        <span class="font-mono text-zinc-300"><span id="jobProgressValue">${progress}</span>%</span>
      </div>
      <div class="progress-track ${indeterminate ? 'indeterminate' : ''}">
        <div class="progress-bar ${barClass}" style="width:${indeterminate ? 38 : progress}%"></div>
      </div>
      <p class="mt-2 text-[12px] text-zinc-400">${esc(job.progress_message || (job.status === 'queued' ? 'aguardando na fila…' : '—'))}</p>
    </div>

    <dl class="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-[11px] sm:grid-cols-4">
      <div><dt class="text-zinc-600">tempo</dt><dd id="jobElapsed" class="font-mono text-zinc-300">${esc(elapsed)}</dd></div>
      <div><dt class="text-zinc-600">tamanho</dt><dd class="font-mono text-zinc-300">${esc(request.width || '—')}×${esc(request.height || '—')}</dd></div>
      <div><dt class="text-zinc-600">imagens</dt><dd class="font-mono text-zinc-300">${images.length} / ${esc(request.num_images || 1)}</dd></div>
      <div><dt class="text-zinc-600">formato</dt><dd class="font-mono text-zinc-300">${esc(request.output_format || 'png')}</dd></div>
      <div><dt class="text-zinc-600">seed</dt><dd class="font-mono text-zinc-300">${esc(request.seed === null || request.seed === undefined ? 'aleatória' : request.seed)}</dd></div>
      <div><dt class="text-zinc-600">provider</dt><dd class="font-mono text-zinc-300">${esc(job.provider || '—')}</dd></div>
      <div class="col-span-2"><dt class="text-zinc-600">modelo</dt><dd class="truncate font-mono text-zinc-300" title="${esc(job.model_id || '')}">${esc(job.model_id || '—')}</dd></div>
    </dl>

    <div class="mt-3 space-y-1 border-t border-ink-700/60 pt-3 text-[11px]">
      <div><span class="text-zinc-600">criado:</span> <span class="font-mono text-zinc-400">${esc(job.created_at || '—')}</span></div>
      <div><span class="text-zinc-600">início:</span> <span class="font-mono text-zinc-400">${esc(job.started_at || '—')}</span></div>
      <div><span class="text-zinc-600">fim:</span> <span class="font-mono text-zinc-400">${esc(job.finished_at || '—')}</span></div>
      ${refs.length ? `<div><span class="text-zinc-600">referências:</span> <span class="font-mono text-zinc-400">${refs.length}</span></div>` : ''}
    </div>

    <details class="mt-3">
      <summary class="cursor-pointer text-[11px] text-zinc-500 hover:text-zinc-300">prompt enviado</summary>
      <p class="mt-2 whitespace-pre-wrap rounded-md border border-ink-700 bg-ink-900/70 p-2 text-[11px] text-zinc-400">${esc(request.prompt || '')}</p>
      ${request.negative_prompt ? `<p class="mt-2 text-[11px] text-zinc-500">negativo: <span class="text-zinc-400">${esc(request.negative_prompt)}</span></p>` : ''}
    </details>

    ${job.error ? `<p class="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-[12px] text-rose-200">${esc(job.error)}</p>` : ''}
  `;
}

function galleryItems() {
  const jobs = state.onlyCurrent && state.currentJobId
    ? sortedJobs().filter((job) => job.job_id === state.currentJobId)
    : sortedJobs();
  const items = [];
  jobs.forEach((job) => {
    // mais novas primeiro: dentro do job a ordem de geração é invertida
    jobImages(job)
      .slice()
      .reverse()
      .forEach((image) => items.push({ image, job }));
  });
  return items;
}

function renderGallery() {
  const grid = $('#gallery');
  const empty = $('#galleryEmpty');
  const items = galleryItems();

  $('#galleryCount').textContent = `${items.length} ${items.length === 1 ? 'imagem' : 'imagens'}`;
  empty.hidden = items.length > 0;

  grid.innerHTML = items
    .map(({ image, job }) => {
      const url = imageUrl(image);
      const name = `${image.image_id}.${image.format || 'png'}`;
      return `
        <figure class="thumb skeleton" data-image-id="${esc(image.image_id)}" data-job-id="${esc(job.job_id)}">
          <img src="${esc(url)}" alt="${esc(`imagem ${shortId(image.image_id)} do job ${shortId(job.job_id)}`)}"
               loading="lazy" decoding="async" data-action="zoom">
          <span class="thumb-badge">#${esc(shortId(job.job_id))}</span>
          <div class="thumb-actions">
            <button type="button" class="thumb-btn" data-action="zoom" title="ampliar" aria-label="ampliar">
              <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="currentColor"><path d="M8 3a5 5 0 1 0 3.1 8.9l3 3a1 1 0 0 0 1.4-1.4l-3-3A5 5 0 0 0 8 3Zm-3 5a3 3 0 1 1 6 0 3 3 0 0 1-6 0Z"/></svg>
            </button>
            <a class="thumb-btn" href="${esc(url)}" download="${esc(name)}" title="baixar" aria-label="baixar">
              <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="currentColor"><path d="M10 2a1 1 0 0 1 1 1v7.6l2.3-2.3a1 1 0 0 1 1.4 1.4l-4 4a1 1 0 0 1-1.4 0l-4-4a1 1 0 0 1 1.4-1.4L9 10.6V3a1 1 0 0 1 1-1ZM4 15a1 1 0 0 1 1 1v1h10v-1a1 1 0 1 1 2 0v1.5A1.5 1.5 0 0 1 15.5 19h-11A1.5 1.5 0 0 1 3 17.5V16a1 1 0 0 1 1-1Z"/></svg>
            </a>
            <button type="button" class="thumb-btn" data-action="delete-image" title="apagar" aria-label="apagar">
              <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="currentColor"><path d="M8 2a1 1 0 0 0-1 1v1H4a1 1 0 0 0 0 2h1l.7 10a2 2 0 0 0 2 1.9h4.6a2 2 0 0 0 2-1.9L15 6h1a1 1 0 1 0 0-2h-3V3a1 1 0 0 0-1-1H8Zm1 4a.8.8 0 0 1 .8.8l.3 7a.8.8 0 0 1-1.6.1l-.3-7A.8.8 0 0 1 9 6Zm2.8.8a.8.8 0 1 1 1.6 0l-.3 7a.8.8 0 0 1-1.6 0l.3-7Z"/></svg>
            </button>
          </div>
          <figcaption class="thumb-meta">${esc(image.width || '?')}×${esc(image.height || '?')} · ${esc(image.format || 'png')} · ${esc(formatBytes(image.bytes))}</figcaption>
        </figure>
      `;
    })
    .join('');
}

function renderHistory() {
  const list = $('#historyList');
  const empty = $('#historyEmpty');
  const jobs = sortedJobs();

  $('#historyCount').textContent = String(jobs.length);
  empty.hidden = jobs.length > 0;

  list.innerHTML = jobs
    .map((job) => {
      const meta = statusMeta(job.status);
      const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
      const request = job.request || {};
      const images = jobImages(job);
      const isCurrent = job.job_id === state.currentJobId;
      const elapsed = formatDuration(jobElapsedSeconds(job));
      const when = parseTs(job.created_at);
      const whenText = when ? new Date(when).toLocaleString('pt-BR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—';

      return `
        <div class="history-item ${isCurrent ? 'is-current' : ''}">
          <button type="button" class="history-main" data-action="select-job" data-job-id="${esc(job.job_id)}">
            <div class="flex items-center gap-2">
              <span class="status-dot ${meta.dot} ${meta.pulse ? 'pulse' : ''}"></span>
              <span class="truncate text-[12px] text-zinc-300" title="${esc(request.prompt || '')}">${esc(request.prompt || '(sem prompt)')}</span>
            </div>
            <div class="mt-1 flex items-center gap-2 text-[10px] text-zinc-500">
              <span class="font-mono">#${esc(shortId(job.job_id))}</span>
              <span>${esc(meta.label)}</span>
              <span class="ml-auto font-mono">${esc(elapsed)}</span>
            </div>
            <div class="progress-mini mt-1.5"><span style="width:${progress}%"></span></div>
            <div class="mt-1 flex items-center gap-2 text-[10px] text-zinc-600">
              <span>${esc(whenText)}</span>
              <span class="font-mono">${esc(request.width || '?')}×${esc(request.height || '?')}</span>
              <span class="ml-auto">${images.length} img</span>
            </div>
          </button>
          <button type="button" class="history-del" data-action="delete-job" data-job-id="${esc(job.job_id)}" title="apagar job" aria-label="apagar job">
            <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="currentColor"><path d="M8 2a1 1 0 0 0-1 1v1H4a1 1 0 0 0 0 2h1l.7 10a2 2 0 0 0 2 1.9h4.6a2 2 0 0 0 2-1.9L15 6h1a1 1 0 1 0 0-2h-3V3a1 1 0 0 0-1-1H8Zm1 4a.8.8 0 0 1 .8.8l.3 7a.8.8 0 0 1-1.6.1l-.3-7A.8.8 0 0 1 9 6Zm2.8.8a.8.8 0 1 1 1.6 0l-.3 7a.8.8 0 0 1-1.6 0l.3-7Z"/></svg>
          </button>
        </div>
      `;
    })
    .join('');
}

/* -------------------------------------------------------------- lightbox */

function openLightbox(jobId, imageId) {
  const job = findJob(jobId);
  if (!job) return;
  const image = jobImages(job).find((img) => img.image_id === imageId);
  if (!image) return;

  state.lightbox = { jobId, imageId };
  const box = $('#lightbox');
  $('#lightboxImg').src = imageUrl(image);
  $('#lightboxImg').alt = `imagem ${shortId(imageId)} do job ${shortId(jobId)}`;
  $('#lightboxMeta').textContent =
    `#${shortId(jobId)} · ${image.width || '?'}×${image.height || '?'} · ${image.format || 'png'} · ${formatBytes(image.bytes)}` +
    (image.seed === null || image.seed === undefined ? '' : ` · seed ${image.seed}`);
  const dl = $('#lightboxDownload');
  dl.href = imageUrl(image);
  dl.setAttribute('download', `${image.image_id}.${image.format || 'png'}`);
  box.hidden = false;
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  state.lightbox = null;
  $('#lightbox').hidden = true;
  $('#lightboxImg').removeAttribute('src');
  document.body.style.overflow = '';
}

function stepLightbox(delta) {
  if (!state.lightbox) return;
  const items = galleryItems();
  if (!items.length) return;
  const index = items.findIndex((item) => item.image.image_id === state.lightbox.imageId);
  if (index < 0) return;
  const next = items[(index + delta + items.length) % items.length];
  openLightbox(next.job.job_id, next.image.image_id);
}

/* ------------------------------------------------------------ wiring */

function bindForm() {
  const form = $('#jobForm');

  form.addEventListener('submit', onSubmit);

  $('#prompt').addEventListener('input', (event) => {
    $('#promptCount').textContent = `${event.target.value.length} / ${PROMPT_MAX}`;
  });

  $('#negativeToggle').addEventListener('click', () => toggleCollapsible('#negativeToggle', '#negativeWrap'));
  $('#advancedToggle').addEventListener('click', () => toggleCollapsible('#advancedToggle', '#advancedWrap'));

  $('#presetRow').addEventListener('click', (event) => {
    const btn = event.target.closest('.preset');
    if (!btn) return;
    const size = Number(btn.dataset.preset);
    $('#width').value = String(size);
    $('#height').value = String(size);
    syncPresetButtons();
    renderSizeHint();
  });

  ['#width', '#height'].forEach((sel) => {
    $(sel).addEventListener('input', () => {
      syncPresetButtons();
      renderSizeHint();
    });
    $(sel).addEventListener('blur', () => {
      const { min, max } = currentSizeBounds();
      const clamped = clampSize(Number($(sel).value), min, max);
      $(sel).value = String(clamped);
      syncPresetButtons();
      renderSizeHint();
    });
  });

  $('#numImages').addEventListener('change', renderSizeHint);
  $('#referenceImages').addEventListener('input', renderSizeHint);

  form.addEventListener('reset', () => {
    setTimeout(() => {
      clearFieldErrors();
      $('#promptCount').textContent = '0 / 4000';
      applyConfigToForm();
    }, 0);
  });
}

function bindGallery() {
  $('#gallery').addEventListener('click', (event) => {
    const figure = event.target.closest('.thumb');
    if (!figure) return;
    const jobId = figure.dataset.jobId;
    const imageId = figure.dataset.imageId;
    const action = event.target.closest('[data-action]');
    const kind = action ? action.dataset.action : null;

    if (kind === 'delete-image') {
      event.preventDefault();
      deleteImage(jobId);
      return;
    }
    if (event.target.closest('a[download]')) return;
    openLightbox(jobId, imageId);
  });

  $('#refreshBtn').addEventListener('click', refreshJobs);
  $('#onlyCurrentToggle').addEventListener('change', (event) => {
    state.onlyCurrent = event.target.checked;
    renderGallery();
  });

  // estado de carregamento (skeleton) e imagem quebrada
  const grid = $('#gallery');
  grid.addEventListener(
    'load',
    (event) => {
      const figure = event.target.closest && event.target.closest('.thumb');
      if (figure) figure.classList.remove('skeleton');
    },
    true
  );
  grid.addEventListener(
    'error',
    (event) => {
      const img = event.target;
      if (!img || img.tagName !== 'IMG') return;
      const figure = img.closest('.thumb');
      if (!figure || figure.classList.contains('broken')) return;
      figure.classList.remove('skeleton');
      figure.classList.add('broken');
      const note = document.createElement('p');
      note.className = 'thumb-note';
      note.textContent = 'imagem indisponível';
      figure.appendChild(note);
    },
    true
  );
}

function bindHistory() {
  $('#historyList').addEventListener('click', (event) => {
    const action = event.target.closest('[data-action]');
    if (!action) return;
    const jobId = action.dataset.jobId;
    if (action.dataset.action === 'select-job') {
      state.currentJobId = jobId;
      renderAll();
      fetchJobOnce(jobId);
      const card = $('#jobCard');
      if (card && card.scrollIntoView) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (action.dataset.action === 'delete-job') {
      deleteJob(jobId);
    }
  });

  $('#historyRefreshBtn').addEventListener('click', refreshJobs);
}

function bindJobCard() {
  $('#jobBody').addEventListener('click', (event) => {
    const action = event.target.closest('[data-action]');
    if (!action || !state.currentJobId) return;
    if (action.dataset.action === 'cancel-job') cancelJob(state.currentJobId);
    if (action.dataset.action === 'delete-job') deleteJob(state.currentJobId);
  });
}

function bindLightbox() {
  $('#lightboxClose').addEventListener('click', closeLightbox);
  $('#lightboxBackdrop').addEventListener('click', closeLightbox);
  $('#lightboxPrev').addEventListener('click', () => stepLightbox(-1));
  $('#lightboxNext').addEventListener('click', () => stepLightbox(1));
  $('#lightboxDelete').addEventListener('click', () => {
    if (!state.lightbox) return;
    deleteImage(state.lightbox.jobId);
  });
  document.addEventListener('keydown', (event) => {
    if ($('#lightbox').hidden) return;
    if (event.key === 'Escape') closeLightbox();
    if (event.key === 'ArrowLeft') stepLightbox(-1);
    if (event.key === 'ArrowRight') stepLightbox(1);
  });
}

/** mantém o "tempo decorrido" andando entre eventos do servidor */
function startTicker() {
  setInterval(() => {
    if (!state.currentJobId) return;
    const job = findJob(state.currentJobId);
    if (!job || isTerminal(job)) return;
    const node = $('#jobElapsed');
    if (node) node.textContent = formatDuration(jobElapsedSeconds(job));
  }, 1000);
}

function bindVisibility() {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    // ao voltar para a aba, ressincroniza o que estiver em andamento
    const active = state.jobs.filter((job) => !isTerminal(job));
    active.forEach((job) => {
      if (!state.watchers.has(job.job_id)) watchJob(job.job_id);
      else fetchJobOnce(job.job_id);
    });
    if (state.currentJobId) fetchJobOnce(state.currentJobId);
  });
}

/* --------------------------------------------------------------- boot */

async function boot() {
  bindForm();
  bindGallery();
  bindHistory();
  bindJobCard();
  bindLightbox();
  bindVisibility();
  startTicker();

  renderAll();
  setInterval(loadHealth, HEALTH_INTERVAL_MS);

  await Promise.all([loadConfig(), loadHealth()]);
  await loadJobs();
}

document.addEventListener('DOMContentLoaded', boot);
window.addEventListener('beforeunload', stopAllWatchers);
