/* Web test client (Nhóm G).
 *
 * Ba việc:
 *   G1 — thu mic, resample về 16kHz mono PCM16, gửi binary qua WebSocket (§4.1)
 *   G2 — hiển thị semantic event realtime
 *   G3 — phát TTS, và DỪNG NGAY khi nhận `tts_cancelled` (Barge-in, §2.4.1)
 *
 * Client cố ý KHÔNG biết gì về cấu trúc JSON của LLM. Nó chỉ hiểu các event
 * ngữ nghĩa: translation_delta / intent_done / reply_ready (§4.4).
 */

const TARGET_SR = 16000;
const CHUNK_MS = 100;

const el = (id) => document.getElementById(id);
const ui = {
  status: el("status"), toggle: el("toggle"), stopTts: el("stopTts"),
  utteranceId: el("utteranceId"), partial: el("partial"), final: el("final"),
  lang: el("lang"), sttLatency: el("sttLatency"),
  translation: el("translation"), intent: el("intent"), replies: el("replies"),
  log: el("log"),
  mE2e: el("mE2e"), mE2eP95: el("mE2eP95"), mTtft: el("mTtft"),
  mStt: el("mStt"), mBarge: el("mBarge"), mUtt: el("mUtt"),
};

const state = {
  ws: null, audioCtx: null, playCtx: null, stream: null, node: null,
  running: false,
  pendingBinary: null,      // metadata của tts_audio_chunk đang chờ frame nhị phân
  playHead: 0,              // mốc lịch phát tiếp theo trong playCtx
  sources: new Set(),       // AudioBufferSourceNode đang phát — để dừng tức thì
  endpointAt: null,         // mốc client thấy stt_final -> xấp xỉ VAD endpoint
  firstUseful: null,
  e2eSamples: [],
  utterances: 0,
  replies: [],
};

/* ------------------------------- nhật ký ------------------------------- */

function log(type, text) {
  const row = document.createElement("div");
  const time = new Date().toLocaleTimeString("vi-VN", { hour12: false });
  let cls = "k";
  if (type.startsWith("tts")) cls += " tts";
  if (type.includes("cancel")) cls += " cancel";
  if (type === "error" || type.includes("error")) cls += " err";
  row.innerHTML = `<span class="t">${time}</span><span class="${cls}">${type}</span> ${text ?? ""}`;
  ui.log.prepend(row);
  while (ui.log.childElementCount > 300) ui.log.lastChild.remove();
}

function setStatus(text, kind) {
  ui.status.textContent = text;
  ui.status.className = `badge ${kind}`;
}

const ms = (v) => (v === null || v === undefined ? "—" : `${Math.round(v)}ms`);

function percentile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q / 100;
  const low = Math.floor(pos);
  const high = Math.min(low + 1, sorted.length - 1);
  return sorted[low] * (1 - (pos - low)) + sorted[high] * (pos - low);
}

/* ------------------------------ thu âm -------------------------------- */

async function startCapture() {
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      // Tắt xử lý của trình duyệt: ta muốn đo đúng thứ mic đưa vào, và các bộ
      // xử lý này che mất chính hiện tượng mà B7/B9 cần quan sát.
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  state.audioCtx = new AudioContext();
  const source = state.audioCtx.createMediaStreamSource(state.stream);
  const inputSr = state.audioCtx.sampleRate;

  const workletCode = `
    class Capture extends AudioWorkletProcessor {
      constructor() { super(); this.buf = []; this.count = 0;
        this.target = Math.round(sampleRate * ${CHUNK_MS} / 1000); }
      process(inputs) {
        const ch = inputs[0][0];
        if (!ch) return true;
        this.buf.push(new Float32Array(ch));
        this.count += ch.length;
        if (this.count >= this.target) {
          const merged = new Float32Array(this.count);
          let off = 0;
          for (const b of this.buf) { merged.set(b, off); off += b.length; }
          this.port.postMessage(merged, [merged.buffer]);
          this.buf = []; this.count = 0;
        }
        return true;
      }
    }
    registerProcessor('capture', Capture);
  `;
  const url = URL.createObjectURL(new Blob([workletCode], { type: "application/javascript" }));
  await state.audioCtx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  state.node = new AudioWorkletNode(state.audioCtx, "capture");
  state.node.port.onmessage = (event) => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    const resampled = resample(event.data, inputSr, TARGET_SR);
    state.ws.send(floatToPcm16(resampled));
  };
  source.connect(state.node);
  // Worklet phải nối tới destination mới được chạy trong một số trình duyệt;
  // gain 0 để không tạo vòng lặp âm thanh ra loa.
  const mute = state.audioCtx.createGain();
  mute.gain.value = 0;
  state.node.connect(mute).connect(state.audioCtx.destination);

  log("mic", `${inputSr}Hz → ${TARGET_SR}Hz, chunk ${CHUNK_MS}ms`);
}

function resample(input, fromSr, toSr) {
  if (fromSr === toSr) return input;
  const ratio = fromSr / toSr;
  const out = new Float32Array(Math.round(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx] ?? 0;
    const b = input[idx + 1] ?? a;
    out[i] = a + (b - a) * frac;   // nội suy tuyến tính
  }
  return out;
}

function floatToPcm16(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out.buffer;
}

/* ------------------------------ phát TTS ------------------------------ */

function playPcmChunk(bytes, sampleRate) {
  if (!state.playCtx) state.playCtx = new AudioContext();
  const ctx = state.playCtx;
  const samples = new Int16Array(bytes);
  if (!samples.length) return;

  const buffer = ctx.createBuffer(1, samples.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);

  const now = ctx.currentTime;
  // Lịch phát nối tiếp nhau. Giữ đệm nhỏ (60ms): đệm càng lớn thì lúc
  // Barge-in càng nhiều audio đã nằm sẵn trong hàng đợi và không hủy được.
  if (state.playHead < now + 0.06) state.playHead = now + 0.06;
  source.start(state.playHead);
  state.playHead += buffer.duration;

  state.sources.add(source);
  source.onended = () => state.sources.delete(source);
}

function stopPlayback() {
  for (const source of state.sources) {
    try { source.stop(); } catch { /* đã dừng rồi */ }
  }
  state.sources.clear();
  state.playHead = 0;
}

/* ---------------------------- xử lý sự kiện --------------------------- */

function onEvent(event) {
  const { type, data, utterance_id: uttId } = event;

  switch (type) {
    case "session_started":
      setStatus(`đã kết nối · ${data.platform}`, "on");
      log(type, `platform=${data.platform} tts=${data.tts_enabled}`);
      break;

    case "audio_started":
      log(type, `${data.sample_rate}Hz`);
      break;

    case "stt_partial":
      ui.partial.textContent = data.text;
      ui.utteranceId.textContent = uttId;
      break;

    case "stt_final": {
      // Xấp xỉ VAD endpoint phía client: thời điểm sớm nhất client biết câu
      // đã dứt. Con số E2E chính xác do server đo (§7); đây là để quan sát.
      state.endpointAt = performance.now() - (data.latency_ms || 0);
      state.firstUseful = null;
      state.utterances += 1;
      ui.mUtt.textContent = state.utterances;

      ui.partial.textContent = "…";
      ui.final.textContent = data.text;
      ui.utteranceId.textContent = uttId;
      ui.lang.textContent = data.language ? `ngôn ngữ: ${data.language}` : "";
      ui.sttLatency.textContent = `STT ${ms(data.latency_ms)}`;
      ui.mStt.textContent = ms(data.latency_ms);
      log(type, `[${data.language}] ${data.text}`);
      break;
    }

    case "copilot_started":
      ui.translation.textContent = "";
      ui.intent.textContent = "";
      ui.replies.innerHTML = "";
      state.replies = [];
      break;

    case "translation_delta":
      ui.translation.textContent = data.full;
      markUseful();
      break;

    case "intent_done":
      ui.intent.textContent = data.intent;
      markUseful();
      log(type, data.intent);
      break;

    case "reply_ready":
      addReply(data.index, data.text);
      markUseful();
      break;

    case "copilot_done":
      ui.mTtft.textContent = ms(data.ttft_ms);
      log(type, `ttft=${ms(data.ttft_ms)} total=${ms(data.total_ms)} tokens=${data.tokens}${data.truncated ? " (BỊ CẮT)" : ""}`);
      break;

    case "tts_started":
      ui.stopTts.disabled = false;
      log(type, `${data.utterance_field}: ${data.text.slice(0, 48)}`);
      break;

    case "tts_audio_chunk":
      state.pendingBinary = data;    // frame nhị phân đến ngay sau
      break;

    case "tts_done":
      ui.stopTts.disabled = true;
      log(type, `${data.chunks} chunk · ${ms(data.synthesis_ms)}`);
      break;

    case "tts_cancelled":
      stopPlayback();                // G3 — dừng NGAY, không đợi hết buffer
      ui.stopTts.disabled = true;
      ui.mBarge.textContent = ms(data.response_ms);
      log(type, `${data.reason} · server ${ms(data.response_ms)} · ${data.chunks_sent} chunk`);
      break;

    case "tts_error":
      ui.stopTts.disabled = true;
      log(type, data.message);
      break;

    case "error":
      log(type, `[${data.code}] ${data.message}`);
      if (!data.recoverable) setStatus("lỗi", "err");
      break;

    case "session_ended":
      log(type, `${data.utterances} utterance`);
      break;
  }
}

function markUseful() {
  if (state.firstUseful !== null || state.endpointAt === null) return;
  state.firstUseful = performance.now();
  const e2e = state.firstUseful - state.endpointAt;
  state.e2eSamples.push(e2e);
  ui.mE2e.textContent = ms(e2e);
  ui.mE2eP95.textContent = ms(percentile(state.e2eSamples, 95));
}

function addReply(index, text) {
  state.replies[index] = text;
  const item = document.createElement("li");
  item.textContent = text;
  item.title = "Bấm để đọc qua tai nghe";
  // §2.4.1 MVP scope: quick reply CHỈ đọc khi người dùng chọn thủ công.
  item.onclick = () => {
    if (state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ action: "speak_reply", reply_index: index, text }));
    }
  };
  ui.replies.append(item);
}

/* ------------------------------ kết nối ------------------------------- */

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws/copilot`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("đã kết nối", "on");

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      onEvent(JSON.parse(event.data));
      return;
    }
    // Frame nhị phân: PCM của tts_audio_chunk vừa được báo trước
    const meta = state.pendingBinary;
    state.pendingBinary = null;
    if (meta) playPcmChunk(event.data, meta.sample_rate);
  };

  ws.onclose = () => { setStatus("mất kết nối", "off"); stop(); };
  ws.onerror = () => setStatus("lỗi kết nối", "err");

  state.ws = ws;
}

async function start() {
  try {
    connect();
    await startCapture();
    state.running = true;
    ui.toggle.textContent = "Dừng";
  } catch (err) {
    setStatus(`lỗi mic: ${err.message}`, "err");
    log("error", err.message);
    stop();
  }
}

function stop() {
  state.running = false;
  ui.toggle.textContent = "Bắt đầu nghe";
  ui.stopTts.disabled = true;
  stopPlayback();
  state.node?.disconnect();
  state.stream?.getTracks().forEach((t) => t.stop());
  state.audioCtx?.close();
  state.audioCtx = null; state.node = null; state.stream = null;
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.close();
}

ui.toggle.onclick = () => (state.running ? stop() : start());
ui.stopTts.onclick = () => {
  stopPlayback();
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action: "cancel_tts" }));
  }
};
