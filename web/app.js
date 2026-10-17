/* 语音增强演示前端：录音 / 上传 -> 本地后端推理 -> 语谱图 + 试听
   所有资源都在本机，无任何外部请求。 */

const TARGET_SR = 16000;
const MAX_SECONDS = 60;

/* ------------------------------------------------------------------
   回声(反馈)抵消配置 —— "不能戴耳机"场景的补救
   外放做实时演示时，麦克风会把扬声器里的增强后语音再录进去，后端对这个
   已经增强过一遍的信号再增强一次，结果就是二次降噪的金属感/发闷。

   把 FEEDBACK.enabled 设为 true，播放时会发出
       (1-g) * 增强输出  +  g * 延迟后的增强输出取反
   即 g * (y(t) - y(t - delayMs)) —— 延迟处的回声被抵消，直接传来的
   人声几乎不受影响。

   delayMs 用项目里的 delay.py 实测：
       python delay.py --ref 参考音频.wav --rec 麦克风录音.wav
   典型外放场景 60~150 ms。调不准就把 gain 调小(0.3~0.5)，宁可少抵一点，
   也不要因为延迟估错而引入梳状滤波的"机器人音"。
   ------------------------------------------------------------------ */
const FEEDBACK = {
  enabled: false,
  delayMs: 0,
  gain: 0.9,
};

const el = (id) => document.getElementById(id);
const state = {
  inputB64: null,
  inputDuration: 0,
  result: null,
  recorder: null,
  recStarting: false,
  stream: null,
  audioCtx: null,
  analyser: null,
  chunks: [],
  recStart: 0,
  recTimer: null,
  rafId: null,
  playNode: null,
  playFilters: null,
};

/* ---------------------------------------------------------- 工具 */

function setStatus(text, kind) {
  const node = el('status');
  node.textContent = text;
  node.className = 'status' + (kind ? ' ' + kind : '');
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}

/* AudioBuffer -> 16k 单声道 WAV(PCM16) -> base64 */
function encodeWav(buffer) {
  const pcm = buffer.getChannelData(0);
  const bytes = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(bytes);
  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + pcm.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, pcm.length * 2, true);
  let off = 44;
  for (let i = 0; i < pcm.length; i++) {
    let v = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    off += 2;
  }
  return bytes;
}

function toBase64(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  let bin = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

function getCtx() {
  if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return state.audioCtx;
}

/* 解码 -> 单声道 -> 重采样 16k */
async function decodeTo16k(arrayBuffer) {
  const ctx = getCtx();
  const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
  const len = Math.max(1, Math.ceil(decoded.duration * TARGET_SR));
  const offline = new OfflineAudioContext(1, len, TARGET_SR);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  return await offline.startRendering();
}

function blobUrlFromB64(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
}

/* 把录音器实际产出的采样率转成 16 kHz，避免误标采样率导致播放变调 */
async function resampleBuffer(src, fromSr, toSr, channels) {
  const ctx = getCtx();
  const len = Math.max(1, Math.round(src.length * toSr / fromSr));
  const offline = new OfflineAudioContext(channels, len, toSr);
  const buf = offline.createBuffer(channels, src.length, fromSr);
  for (let c = 0; c < channels; c++) {
    buf.copyToChannel(src.getChannelData(Math.min(c, src.numberOfChannels - 1)), c);
  }
  const node = offline.createBufferSource();
  node.buffer = buf;
  node.connect(offline.destination);
  node.start();
  return await offline.startRendering();
}

/* ---------------------------------------------------------- 回声抵消播放 */

/* 把增强输出处理成  g * (y(t) - y(t-delay))  后送进扬声器 */
async function applyFeedbackCancel(buffer) {
  const d = Math.round(FEEDBACK.delayMs / 1000 * TARGET_SR);
  if (d <= 0) return buffer;
  const n = buffer.length;
  const pcm = buffer.getChannelData(0);
  const out = new Float32Array(n);
  const g = FEEDBACK.gain;
  for (let i = 0; i < n; i++) {
    out[i] = pcm[i] - (i >= d ? g * pcm[i - d] : 0);
  }
  const ctx = getCtx();
  const offline = new OfflineAudioContext(1, n, TARGET_SR);
  const buf = offline.createBuffer(1, n, TARGET_SR);
  buf.copyToChannel(out, 0);
  const node = offline.createBufferSource();
  node.buffer = buf;
  node.connect(offline.destination);
  node.start();
  return await offline.startRendering();
}

async function makePlayBuffer(b64) {
  const buf = await getCtx().decodeAudioData(
    await (await fetch(blobUrlFromB64(b64))).arrayBuffer()
  );
  if (!FEEDBACK.enabled) return buf;
  return applyFeedbackCancel(buf);
}

function stopPlayback() {
  if (state.playNode) {
    try { state.playNode.stop(); } catch (e) { /* 已经停了 */ }
    state.playNode = null;
  }
}

/* 用 WebAudio 播放并带上回声抵消；任何一步失败都退回原生 <audio> */
async function playEnhanced() {
  if (!state.result) return;
  const btn = el('playEnhanced');
  if (state.playNode) { stopPlayback(); btn.textContent = '试听增强输出'; return; }
  if (!FEEDBACK.enabled) {
    el('outEnhanced').play();
    return;
  }
  btn.disabled = true;
  try {
    const ctx = getCtx();
    const buf = await makePlayBuffer(state.result.audio);
    const node = ctx.createBufferSource();
    node.buffer = buf;
    node.connect(ctx.destination);
    node.onended = () => { state.playNode = null; btn.textContent = '试听增强输出'; };
    node.start();
    state.playNode = node;
    btn.textContent = '停止播放';
  } catch (err) {
    setStatus('回声抵消播放失败，已退回普通播放：' + err.message, 'err');
    el('outEnhanced').play();
  } finally {
    btn.disabled = false;
  }
}

/* ---------------------------------------------------------- 输入录入 */

async function adoptAudio(arrayBuffer, label, duration) {
  const buf = await decodeTo16k(arrayBuffer);
  if (buf.duration > MAX_SECONDS) {
    setStatus(`音频 ${buf.duration.toFixed(1)}s 超过 ${MAX_SECONDS}s 上限，请剪短后再试`, 'err');
    return;
  }

  // 纯静音/极安静录音提前拦下：这类音频没有有效语音，后端算不出有意义的
  // 指标（比值的分子分母都趋近 0），提前给提示比让它跑完再报错要好。
  const pcm = buf.getChannelData(0);
  let peak = 0, sumSq = 0;
  for (let i = 0; i < pcm.length; i++) {
    const a = Math.abs(pcm[i]);
    if (a > peak) peak = a;
    sumSq += pcm[i] * pcm[i];
  }
  const rms = Math.sqrt(sumSq / Math.max(1, pcm.length));
  if (peak < 1e-4 || rms < 1e-5) {
    setStatus('这段音频几乎是静音，没有可增强的语音。请确认麦克风选对了、系统没静音，再录一次。', 'err');
    return;
  }

  state.inputB64 = toBase64(encodeWav(buf));
  state.inputDuration = buf.duration;
  el('inputMeta').textContent = `${label} · ${buf.duration.toFixed(2)} s · 16 kHz 单声道`;
  el('inputAudio').src = blobUrlFromB64(state.inputB64);
  drawWave(el('inputWave'), buf.getChannelData(0), '#8b949e');
  el('enhanceBtn').disabled = false;
  setStatus('已就绪，点击“开始增强”', 'ok');
}

/* 录音 */
async function toggleRecord() {
  if (state.recorder && state.recorder.state === 'recording') {
    stopRecord();
    return;
  }
  // getUserMedia 是异步的，弹出权限框期间用户可能又点了一下。
  // 没有这个锁的话会并发申请两次麦克风，产生两个 recorder。
  if (state.recStarting) return;
  state.recStarting = true;
  el('recBtn').disabled = true;
  try {
    // 关掉浏览器自带降噪，否则演示的是浏览器而不是我们的模型
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (err) {
    setStatus('无法访问麦克风：' + err.message, 'err');
    el('recBtn').disabled = false;
    state.recStarting = false;
    return;
  } finally {
    state.recStarting = false;
  }
  el('recBtn').disabled = false;

  const ctx = getCtx();
  const src = ctx.createMediaStreamSource(state.stream);
  state.analyser = ctx.createAnalyser();
  state.analyser.fftSize = 1024;
  src.connect(state.analyser);

  state.chunks = [];
  try {
    state.recorder = new MediaRecorder(state.stream);
  } catch (err) {
    // 个别浏览器/设备组合不支持 MediaRecorder，别让按钮卡在"停止录音"
    state.stream.getTracks().forEach((t) => t.stop());
    resetRecordUI();
    setStatus('这个浏览器无法录音（MediaRecorder 不可用）：' + err.message, 'err');
    return;
  }
  state.recorder.ondataavailable = (e) => { if (e.data.size) state.chunks.push(e.data); };
  state.recorder.onerror = (e) => {
    setStatus('录音出错：' + (e.error ? e.error.name : '未知错误'), 'err');
    resetRecordUI();
  };
  state.recorder.onstop = async () => {
    const blob = new Blob(state.chunks, { type: state.recorder.mimeType || 'audio/webm' });
    state.stream.getTracks().forEach((t) => t.stop());
    resetRecordUI();               // 先让按钮/计时器/动画全部复位，再做耗时的解码
    el('recHint').textContent = '正在处理录音…';
    try {
      await adoptAudio(await blob.arrayBuffer(), '麦克风录音', 0);
    } catch (err) {
      setStatus('录音解码失败：' + err.message, 'err');
    }
    el('recHint').textContent = '点击后允许麦克风权限，说完再点停止';
  };
  state.recStart = Date.now();
  el('recBtn').textContent = '停止录音';
  el('recBtn').classList.add('recording');
  el('recTime').textContent = '00:00';

  try {
    state.recorder.start();
  } catch (err) {
    state.stream.getTracks().forEach((t) => t.stop());
    resetRecordUI();
    setStatus('无法开始录音：' + err.message, 'err');
    return;
  }

  state.recTimer = setInterval(() => {
    const sec = (Date.now() - state.recStart) / 1000;
    el('recTime').textContent = fmtTime(sec);
    if (sec >= MAX_SECONDS) stopRecord();
  }, 200);
  drawLive();
}

/* 录音结束后把按钮、计时器、动画全部复位。
   之前这里只恢复了计时器和动画，漏了按钮文案和样式类，
   于是点完"停止录音"按钮一直显示"停止录音"。 */
function resetRecordUI() {
  cancelAnimationFrame(state.rafId);
  state.rafId = null;
  clearInterval(state.recTimer);
  state.recTimer = null;
  const btn = el('recBtn');
  btn.textContent = '开始录音';
  btn.classList.remove('recording');
  btn.disabled = false;
}

function stopRecord() {
  if (state.recorder && state.recorder.state === 'recording') {
    state.recorder.stop();
  } else {
    // 录音器已经不在 recording 状态（比如权限被撤销、设备被拔出），
    // 此时 stop() 不会再触发 onstop，手动复位按钮，避免卡在"停止录音"。
    resetRecordUI();
  }
}

function drawLive() {
  const canvas = el('liveCanvas');
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(state.analyser.fftSize);
  const loop = () => {
    state.rafId = requestAnimationFrame(loop);
    state.analyser.getByteTimeDomainData(buf);
    const w = canvas.clientWidth;
    if (canvas.width !== w) canvas.width = w;
    ctx.fillStyle = '#1c222b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    const step = buf.length / canvas.width;
    for (let i = 0; i < canvas.width; i++) {
      const v = buf[Math.floor(i * step)] / 128.0;
      const y = (v * canvas.height) / 2;
      if (i === 0) ctx.moveTo(i, y); else ctx.lineTo(i, y);
    }
    ctx.strokeStyle = '#3fb27f';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };
  loop();
}

/* ---------------------------------------------------------- 绘制 */

function drawWave(canvas, data, color) {
  const w = canvas.clientWidth || 600;
  canvas.width = w;
  canvas.height = 72;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1c222b';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const step = Math.max(1, Math.floor(data.length / w));
  ctx.beginPath();
  for (let i = 0; i < w; i++) {
    let mx = 0;
    for (let j = 0; j < step; j++) {
      const v = data[i * step + j] || 0;
      if (Math.abs(v) > mx) mx = Math.abs(v);
    }
    const h = mx * canvas.height * 0.9;
    ctx.rect(i, (canvas.height - h) / 2, 1, Math.max(h, 1));
  }
  ctx.fillStyle = color;
  ctx.fill();
}

const STOPS = [
  [8, 10, 24], [22, 42, 92], [26, 96, 132],
  [46, 160, 130], [150, 196, 92], [246, 226, 122], [255, 250, 210],
];

function colorAt(v) {
  const t = (v / 255) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t));
  const f = t - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

function drawSpec(canvas, grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    const src = grid[rows - 1 - r];   // 低频画在下方
    for (let c = 0; c < cols; c++) {
      const [R, G, B] = colorAt(src[c]);
      const i = (r * cols + c) * 4;
      img.data[i] = R; img.data[i + 1] = G; img.data[i + 2] = B; img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/* ---------------------------------------------------------- 推理 */

async function runEnhance() {
  if (!state.inputB64) return;
  el('enhanceBtn').disabled = true;
  setStatus('模型推理中，请稍候…', 'busy');
  try {
    const resp = await fetch('/api/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio: state.inputB64, sr: TARGET_SR }),
    });

    // 先拿文本再自己解析，而不是直接 resp.json()。
    // 这样解析失败时我们能拿到响应体的真实片段，给出有用的提示；
    // resp.json() 抛出的 "No number after minus sign in JSON at position N"
    // 这类底层报错既看不懂，也定位不到原因。
    const text = await resp.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (parseErr) {
      // 后端偶发非有限数值（NaN / Infinity）时会写出非法 JSON，
      // 这里识别出来并给出明确解释，而不是把原始报错甩给用户。
      const bad = /-?Infinity|NaN/.exec(text);
      if (bad) {
        throw new Error(
          `后端返回了非法 JSON 字面量 ${bad[0]}（位置 ${bad.index}）。` +
          `这通常意味着某个指标算出了无穷大或 NaN，请把后端终端日志发出来。`
        );
      }
      throw new Error(
        `响应不是合法 JSON（共 ${text.length} 字符）：${parseErr.message}。` +
        `若是响应被截断，请查看后端终端是否完整打印了"已返回给学生"。`
      );
    }
    if (!data.ok) throw new Error(data.error || '推理失败');

    // 数值字段兜底：任何一项不是有限数就退回占位符，避免 .toFixed 崩掉整页
    const num = (v, digits, fallback = '—') => {
      const n = Number(v);
      return Number.isFinite(n) ? n.toFixed(digits) : fallback;
    };
    const m = data.metrics || {};
    if (!Array.isArray(data.spec_before) || !Array.isArray(data.spec_after)) {
      throw new Error('响应里缺少语谱图矩阵');
    }

    state.result = data;
    drawSpec(el('specBefore'), data.spec_before);
    drawSpec(el('specAfter'), data.spec_after);
    el('durLabel1').textContent = num(data.duration, 2, '—') + ' s';
    el('durLabel2').textContent = num(data.duration, 2, '—') + ' s';

    el('outInput').src = blobUrlFromB64(state.inputB64);
    el('outEnhanced').src = blobUrlFromB64(data.audio);

    el('mNoise').textContent = num(m.noise_reduction_db, 1);
    el('mSpeech').textContent = num(m.speech_change_db, 1);
    el('mCost').textContent = num(data.process_ms, 0);
    el('mRtf').textContent = num(data.rtf, 3);
    el('mDur').textContent = num(data.duration, 2);

    const sc = Number(m.speech_change_db);
    let tip = '语音保真接近 0 dB 说明人声基本没被削掉。';
    if (Number.isFinite(sc) && sc < -3) {
      tip = '语音保真明显为负，模型把一部分人声也削掉了（过度抑制），可以换个 checkpoint 试试。';
    } else if (!Number.isFinite(sc)) {
      tip = '这条音频几乎没有有效语音段（可能是静音），指标不可靠，请录一段有说话的音频再试。';
    }
    el('metricTip').textContent = tip;

    el('downloadBtn').disabled = false;
    el('playEnhanced').disabled = false;
    setStatus('完成，可以对比试听', 'ok');
  } catch (err) {
    setStatus('失败：' + err.message, 'err');
  } finally {
    el('enhanceBtn').disabled = false;
  }
}

/* ---------------------------------------------------------- 事件绑定 */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.tabbody').forEach((b) => b.classList.remove('active'));
    tab.classList.add('active');
    el('tab-' + tab.dataset.tab).classList.add('active');
  });
});

el('recBtn').addEventListener('click', toggleRecord);

const dropZone = el('dropZone');
const fileInput = el('fileInput');
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropZone.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (file) await handleFile(file);
});
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) await handleFile(file);
});

async function handleFile(file) {
  el('fileMeta').textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  setStatus('正在解码音频…', 'busy');
  await adoptAudio(await file.arrayBuffer(), file.name, 0);
}

el('enhanceBtn').addEventListener('click', runEnhance);

el('playEnhanced').addEventListener('click', playEnhanced);

el('downloadBtn').addEventListener('click', () => {
  if (!state.result) return;
  const a = document.createElement('a');
  a.href = blobUrlFromB64(state.result.audio);
  a.download = 'enhanced.wav';
  a.click();
});

window.addEventListener('resize', () => {
  if (state.result) {
    drawSpec(el('specBefore'), state.result.spec_before);
    drawSpec(el('specAfter'), state.result.spec_after);
  }
});

/* 无耳机提示：开了回声抵消才提醒，避免默认状态就弹一条没用的警告 */
function renderFeedbackHint() {
  const hint = el('feedbackHint');
  if (!FEEDBACK.enabled) {
    hint.textContent = '实时试听建议戴耳机；不戴耳机时增强输出会被麦克风再次录入，听起来像"二次降噪"。';
    return;
  }
  if (FEEDBACK.delayMs > 0) {
    hint.textContent = `已开启回声抵消（延迟 ${FEEDBACK.delayMs} ms，增益 ${FEEDBACK.gain}）。`;
  } else {
    hint.textContent = '回声抵消已开启但 delayMs 还是 0，请先用 delay.py 实测延迟，否则抵消无效。';
  }
}

/* 启动时拉一下模型信息 */
fetch('/api/info')
  .then((r) => r.json())
  .then((info) => {
    const m = info.model || {};
    el('modelInfo').innerHTML =
      `${(m.ckpt || '').split(/[\\/]/).pop()}<br>` +
      `epoch ${m.epoch ?? '?'} · ${(m.params / 1e6).toFixed(2)}M 参数 · ` +
      `best SI-SNRi ${m.best_si_snri ? m.best_si_snri.toFixed(2) + ' dB' : '?'} · ${m.sr / 1000} kHz`;
  })
  .catch(() => { el('modelInfo').textContent = '未连接到后端'; });

renderFeedbackHint();
