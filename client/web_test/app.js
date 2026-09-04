/* Web test client (Nhóm G).
 *
 * Ba việc:
 *   G1 — thu mic, resample về 16kHz mono PCM16, gửi binary qua WebSocket (§4.1)
 *   G2 — hiển thị semantic event realtime
 *   G3 — phát TTS, và DỪNG NGAY khi nhận `tts_cancelled` (Barge-in, §2.4.1)
 *
 * Client cố ý KHÔNG biết gì về cấu trúc JSON của LLM. Nó chỉ hiểu các event
 * ngữ nghĩa: translation_delta (§4.4).
 */

const TARGET_SR = 16000;
const CHUNK_MS = 100;

const el = (id) => document.getElementById(id);
const ui = {
  status: el("status"), toggle: el("toggle"), stopTts: el("stopTts"),
  utteranceId: el("utteranceId"), partial: el("partial"), final: el("final"),
  lang: el("lang"), sttLatency: el("sttLatency"),
  translation: el("translation"), translationTitle: el("translationTitle"),
  coachHint: el("coachHint"),
  log: el("log"), pickFile: el("pickFile"), fileInput: el("fileInput"),
  pauseBtn: el("pauseBtn"), autoPause: el("autoPause"),
  autoPauseWrap: el("autoPauseWrap"), filePanel: el("filePanel"),
  fileName: el("fileName"), fileBar: el("fileBar"), filePos: el("filePos"),
  fileState: el("fileState"), reviewSteps: el("reviewSteps"),
  mE2e: el("mE2e"), mE2eP95: el("mE2eP95"), mTtft: el("mTtft"),
  mStt: el("mStt"), mBarge: el("mBarge"), mUtt: el("mUtt"),
};

const state = {
  ws: null, audioCtx: null, playCtx: null, stream: null, node: null,
  running: false,
  pendingBinary: null,      // metadata của tts_audio_chunk đang chờ frame nhị phân
  playHead: 0,              // mốc lịch phát tiếp theo trong playCtx
  sources: new Set(),       // AudioBufferSourceNode của TTS — để dừng tức thì
  fileSources: new Set(),   // ... của file đang phát. Tách riêng khỏi TTS:
                            // hủy TTS không được làm câm file, và dừng file
                            // không được cắt lời đang đọc dở.
  endpointAt: null,         // mốc client thấy stt_final -> xấp xỉ VAD endpoint
  firstUseful: null,
  e2eSamples: [],
  utterances: 0,
  file: null,        // bộ phát file, xem streamFile()
  ttsEnabled: true,  // server báo qua session_started
  utt: null,         // dữ liệu câu đang xử lý, gom để nghe lại
  review: null,      // tiến trình nghe lại đang chạy
  ttsWaiter: null,   // resolve khi lượt TTS hiện tại kết thúc
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

/* --------------------------- phát tiếng file -------------------------- */

function ensurePlayCtx() {
  if (!state.playCtx) state.playCtx = new AudioContext();
  return state.playCtx;
}

/** Lên lịch phát một chunk file tại đúng mốc `at` trong đồng hồ AudioContext.
 *
 * File PHẢI phát ra loa: người dùng cần nghe câu gốc lúc nó chạy tới, rồi mới
 * tới bản dịch. Trước đây playbackLoop chỉ GỬI PCM lên server, không phát gì
 * cả — câu gốc chỉ nghe được ở bước phát lại.
 */
function scheduleFileChunk(chunk, at) {
  const ctx = ensurePlayCtx();
  const buffer = ctx.createBuffer(1, chunk.length, TARGET_SR);
  buffer.getChannelData(0).set(chunk);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(at);
  state.fileSources.add(source);
  source.onended = () => state.fileSources.delete(source);
}

function stopFileAudio() {
  for (const source of state.fileSources) {
    try { source.stop(); } catch { /* đã dừng rồi */ }
  }
  state.fileSources.clear();
}

/* ---------------------------- xử lý sự kiện --------------------------- */

function onEvent(event) {
  const { type, data, utterance_id: uttId } = event;
  noteBusy(type);

  switch (type) {
    case "session_started":
      state.ttsEnabled = data.tts_enabled;
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

    case "utterance_endpoint":
      // VAD vừa chốt câu -> dừng file NGAY tại đây. Đợi `stt_final` thì đã
      // phát lấn sang câu sau ~1.9s (thời gian Whisper nghe).
      beginUtteranceHold();
      log(type, `${data.trigger} · ${data.duration_s}s`);
      break;

    case "utterance_continued":
      // Server nghe ra câu còn dở, đang chờ nói tiếp. PHẢI phát tiếp: file
      // đang dừng ở utterance_endpoint, không phát thì audio cần để quyết
      // định sẽ không bao giờ tới và nó dừng vĩnh viễn.
      ui.fileState.textContent = "câu chưa hết — phát tiếp";
      log(type, `chưa trọn câu: ${JSON.stringify(data.text)}`);
      resumePlayback();
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
      applyDirection(data.direction);
      ui.lang.textContent = data.language ? `ngôn ngữ: ${data.language}` : "";
      ui.sttLatency.textContent = `STT ${ms(data.latency_ms)}`;
      ui.mStt.textContent = ms(data.latency_ms);
      log(type, `[${data.language}] ${data.text}`);

      // Server đã chốt được một câu -> giữ file lại cho tới khi nghe xong.
      state.utt = {
        id: uttId,
        startS: data.start_s ?? 0,
        durationS: data.duration_s ?? 0,
        source: data.text,
        direction: data.direction || "to_user",
        translation: "",
      };
      updateFileUi();
      break;
    }

    case "copilot_started":
      ui.translation.textContent = "";
      break;

    case "translation_delta":
      ui.translation.textContent = data.full;
      applyDirection(data.direction);
      if (state.utt) state.utt.translation = data.full;
      markUseful();
      break;


    case "copilot_done":
      ui.mTtft.textContent = ms(data.ttft_ms);
      log(type, `ttft=${ms(data.ttft_ms)} total=${ms(data.total_ms)} tokens=${data.tokens}${data.truncated ? " (BỊ CẮT)" : ""}`);
      maybeStartReview();
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

    case "utterance_dropped":
      // Người dùng PHẢI biết mình vừa mất một câu — im lặng thì họ tưởng
      // đối phương không nói gì.
      setStatus("bỏ 1 câu do xử lý không kịp", "warn");
      log(type, `bỏ câu ${uttId} (${data.reason}, hàng đợi ${data.pending})`);
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

/* Hai chiều: đối phương nói -> đây là bản dịch để HIỂU;
 * bạn nói -> đây là câu tiếng đối phương để NÓI THEO.
 * Nhãn phải đổi, nếu không người dùng không biết mình đang nhìn cái gì.     */
function applyDirection(direction) {
  if (!direction) return;
  const outbound = direction === "to_counterpart";
  ui.translationTitle.textContent = outbound ? "Bạn nói — hãy đọc theo" : "Bản dịch";
  ui.coachHint.hidden = !outbound;
  ui.translation.classList.toggle("coach", outbound);
}

function markUseful() {
  if (state.firstUseful !== null || state.endpointAt === null) return;
  state.firstUseful = performance.now();
  const e2e = state.firstUseful - state.endpointAt;
  state.e2eSamples.push(e2e);
  ui.mE2e.textContent = ms(e2e);
  ui.mE2eP95.textContent = ms(percentile(state.e2eSamples, 95));
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
    state.file = null;         // chế độ micro: không có gì để tạm dừng
    updateFileUi();
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
  state.review = null;
  state.utt = null;
  if (state.ttsWaiter) { state.ttsWaiter(); state.ttsWaiter = null; }
  ui.reviewSteps.hidden = true;
  if (state.file) {
    const wake = state.file.wake;
    state.file = null;         // playbackLoop thấy khác tham chiếu thì tự thoát
    if (wake) wake();
  }
  updateFileUi();
  ui.toggle.textContent = "Bắt đầu nghe";
  ui.stopTts.disabled = true;
  stopPlayback();
  stopFileAudio();
  state.node?.disconnect();
  state.stream?.getTracks().forEach((t) => t.stop());
  state.audioCtx?.close();
  state.audioCtx = null; state.node = null; state.stream = null;
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.close();
}

/* ------------------- phát file thay cho micro ------------------------- */

/* Vì sao cần: sản phẩm này nghe NGƯỜI ĐỐI DIỆN nói ngoại ngữ. Muốn thử bằng
 * micro thì phải có người nói tiếng Anh với bạn. Chế độ này phát một file
 * audio qua đúng đường mà micro đi — cùng resample, cùng chunk 100ms, cùng
 * WebSocket — nên nó kiểm chứng đúng pipeline thật, không phải đường tắt.
 *
 * Tạm dừng an toàn nhờ một tính chất của kiến trúc: VAD phía server chạy theo
 * FRAME, không theo đồng hồ thực. Ngừng gửi audio thì trạng thái VAD đóng băng
 * nguyên vẹn; gửi tiếp là nó chạy đúng chỗ cũ. Không mất câu, không lệch biên. */

function updateFileUi() {
  const f = state.file;
  if (!f) {
    ui.filePanel.hidden = true;
    ui.pauseBtn.hidden = true;
    ui.autoPauseWrap.hidden = true;
    return;
  }
  ui.filePanel.hidden = false;
  ui.pauseBtn.hidden = false;
  ui.autoPauseWrap.hidden = false;
  ui.fileName.textContent = f.name;
  ui.pauseBtn.textContent = f.paused ? "Tiếp tục" : "Tạm dừng";

  const done = Math.min(f.index, f.chunks.length);
  const pct = (done / f.chunks.length) * 100;
  ui.fileBar.style.width = `${pct}%`;
  ui.fileBar.classList.toggle("held", f.paused);

  const at = (done * CHUNK_MS) / 1000;
  const total = (f.chunks.length * CHUNK_MS) / 1000;
  ui.filePos.textContent = `${at.toFixed(1)}s / ${total.toFixed(1)}s · câu ${state.utterances}`;
  ui.fileState.textContent = f.finished
    ? "đã phát hết"
    : f.paused
      ? (f.heldReason || "đang tạm dừng")
      : "đang phát";
}

function pausePlayback(reason) {
  const f = state.file;
  if (!f || f.paused || f.finished) return;
  f.paused = true;
  f.heldReason = reason || "";
  // Cắt luôn phần đã lên lịch trước, nếu không loa còn kêu thêm ~0.2s sau khi
  // đã "dừng" — chồng lên đầu bản dịch sắp đọc.
  stopFileAudio();
  updateFileUi();
}

function resumePlayback() {
  const f = state.file;
  if (!f || !f.paused) return;
  f.paused = false;
  f.heldReason = "";
  const resolve = f.wake;
  f.wake = null;
  if (resolve) resolve();
  updateFileUi();
}

/* Đánh thức chuỗi nghe lại khi một lượt TTS kết thúc, dù vì lý do gì.
 *
 * Không đợi riêng `tts_done`: nếu TTS lỗi hoặc bị hủy mà ta vẫn đợi thì chuỗi
 * nghe lại treo và file không bao giờ phát tiếp.                             */
function noteBusy(type) {
  if (state.ttsWaiter && (type === "tts_done" || type === "tts_cancelled" || type === "tts_error")) {
    const resolve = state.ttsWaiter;
    state.ttsWaiter = null;
    resolve();
    return;
  }
  // Pipeline lỗi giữa chừng thì sẽ không có chuỗi nghe lại nào chạy — phát
  // tiếp để không kẹt vĩnh viễn ở một câu hỏng.
  if (type === "error" && state.file?.paused && !state.review) {
    log("file", "câu lỗi — bỏ qua, phát tiếp");
    resumePlayback();
  }
}

/* ------------------ nghe lại một câu, theo thứ tự ---------------------- */

/* Bốn bước: âm thanh GỐC -> bản dịch -> hàm ý -> gợi ý trả lời.
 *
 * Server ở chế độ `manual` nên nó không tự đọc gì; client quyết thứ tự. Nếu
 * để server tự đọc theo §2.4.1 thì bản dịch sẽ phát TRƯỚC cả âm thanh gốc,
 * vì streaming TTS bắt đầu ngay khi có câu đầu tiên.
 *
 * Giọng đổi theo từng đoạn: khung dẫn và hàm ý đọc giọng Việt, còn câu gợi ý
 * trả lời đọc giọng Anh — đó là ngôn ngữ người dùng sẽ nói ra. Một giọng đọc
 * cả hai thì phần tiếng Anh nghe rất khó hiểu.                              */

function markStep(step, cls) {
  ui.reviewSteps.hidden = false;
  for (const li of ui.reviewSteps.children) {
    if (li.dataset.step !== step) continue;
    li.classList.remove("active", "done", "skip");
    if (cls) li.classList.add(cls);
  }
}

function resetSteps() {
  for (const li of ui.reviewSteps.children) li.classList.remove("active", "done", "skip");
}


function speakAndWait(text, field) {
  if (!text?.trim() || state.ws?.readyState !== WebSocket.OPEN) return Promise.resolve();
  return new Promise((resolve) => {
    state.ttsWaiter = resolve;
    state.ws.send(JSON.stringify({ action: "speak", text, field }));
    // Không để treo vĩnh viễn nếu TTS hỏng mà không phát event nào.
    setTimeout(() => {
      if (state.ttsWaiter === resolve) { state.ttsWaiter = null; resolve(); }
    }, 30000);
  });
}

async function runReview(utt) {
  state.review = utt;
  resetSteps();
  ui.reviewSteps.hidden = false;

  try {
    // KHÔNG phát lại âm thanh gốc: file đã phát tới hết câu này rồi mới dừng,
    // nghe lại là thừa. Dừng ở đâu thì đọc bản dịch ngay ở đó.
    if (state.review !== utt) return;
    const outbound = utt.direction === "to_counterpart";
    const canSpeak = state.ttsEnabled && utt.translation;
    markStep("translation", canSpeak ? "active" : "skip");
    if (canSpeak) {
      ui.fileState.textContent = outbound
        ? "nghe lại: câu để nói theo (đọc chậm)"
        : "nghe lại: bản dịch";
      // Chiều nào thì server tự chọn giọng và tốc độ — client chỉ nói đọc cái gì.
      await speakAndWait(utt.translation, "translation");
      markStep("translation", "done");
    }
  } finally {
    if (state.review === utt) {
      state.review = null;
      ui.fileState.textContent = "đọc xong — phát tiếp";
      setTimeout(resumePlayback, 300);
    }
  }
}

function maybeStartReview() {
  // Chỉ chạy ở chế độ nghe lại từng câu khi đang phát file.
  if (!state.file || !ui.autoPause.checked || !state.utt) return;
  if (state.review) return;
  runReview(state.utt);
}

function beginUtteranceHold() {
  if (!state.file || !ui.autoPause.checked) return;
  // Dừng đúng chỗ câu vừa dứt, rồi mới đọc bản dịch của chính câu đó. Không
  // để file chạy tiếp trong lúc đang đọc: nghe hai câu chồng nhau thì loạn.
  // runReview() chịu trách nhiệm phát tiếp sau khi đọc xong.
  pausePlayback("đang chờ bản dịch câu này");
}

async function streamFile(file) {
  if (state.running) stop();
  connect();
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("hết giờ chờ kết nối")), 8000);
    state.ws.addEventListener("open", () => { clearTimeout(timer); resolve(); }, { once: true });
  });

  const ctx = new AudioContext({ sampleRate: TARGET_SR });
  const decoded = await ctx.decodeAudioData(await file.arrayBuffer());
  await ctx.close();

  // Trộn về mono: micro là mono, và VAD/Whisper đều mong đợi 1 kênh.
  const channels = decoded.numberOfChannels;
  const mono = new Float32Array(decoded.length);
  for (let c = 0; c < channels; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += data[i] / channels;
  }
  const pcm = resample(mono, decoded.sampleRate, TARGET_SR);

  // Chèn im lặng ở hai đầu: VAD cần nền im lặng để chốt được đầu và cuối câu.
  const pad = new Float32Array(TARGET_SR * 0.5);
  const tail = new Float32Array(TARGET_SR * 1.2);
  const full = new Float32Array(pad.length + pcm.length + tail.length);
  full.set(pad, 0);
  full.set(pcm, pad.length);
  full.set(tail, pad.length + pcm.length);

  const step = Math.round((TARGET_SR * CHUNK_MS) / 1000);
  const chunks = [];
  for (let i = 0; i < full.length; i += step) chunks.push(full.subarray(i, i + step));

  state.file = {
    name: file.name, chunks, index: 0, pcm: full,
    paused: false, finished: false, heldReason: "", wake: null,
  };
  state.utt = null;
  state.review = null;

  // Chế độ nghe lại: client quyết thứ tự đọc, server không tự chen vào.
  // Nếu để server tự đọc theo §2.4.1 thì bản dịch sẽ phát TRƯỚC âm thanh gốc.
  setTtsMode(ui.autoPause.checked ? "manual" : "auto");
  state.running = true;
  ui.toggle.textContent = "Dừng";
  setStatus("đang phát file", "on");
  log("file", `${file.name} · ${decoded.sampleRate}Hz ${channels}ch · ${decoded.duration.toFixed(1)}s`);
  updateFileUi();

  await playbackLoop();
}

async function playbackLoop() {
  const f = state.file;
  const ctx = ensurePlayCtx();
  const chunkS = CHUNK_MS / 1000;
  // Lên lịch trước ngần này để tiếng không bị vấp, nhưng đủ nhỏ để lúc dừng
  // ở cuối câu thì loa im gần như tức thì.
  const LEAD_S = 0.2;
  f.head = ctx.currentTime + 0.15;

  while (f === state.file && f.index < f.chunks.length) {
    if (f.paused) {
      // Chờ tới khi được tiếp tục. Không bận rộn quay vòng.
      await new Promise((resolve) => { f.wake = resolve; });
      // Phát tiếp từ bây giờ, không phải từ mốc lịch cũ đã trôi qua.
      f.head = ctx.currentTime + 0.15;
      continue;
    }
    if (!state.running || state.ws?.readyState !== WebSocket.OPEN) return;

    const chunk = f.chunks[f.index];
    state.ws.send(floatToPcm16(chunk));
    scheduleFileChunk(chunk, f.head);   // nghe được đúng đoạn vừa gửi
    f.head += chunkS;
    f.index += 1;
    if (f.index % 5 === 0 || f.index === f.chunks.length) updateFileUi();

    // Nhịp lấy theo ĐỒNG HỒ AUDIO, không phải setTimeout: setTimeout trôi vài
    // phần trăm mỗi nhịp, sau 20 giây là lệch cả giây giữa cái nghe được và
    // cái đã gửi lên server — đúng thứ khiến "dừng ở cuối câu" dừng sai chỗ.
    const wait = (f.head - ctx.currentTime - LEAD_S) * 1000;
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
  if (f === state.file) {
    f.finished = true;
    updateFileUi();
    setStatus("phát xong — chờ kết quả", "on");
  }
}

ui.pickFile.onclick = () => ui.fileInput.click();
ui.fileInput.onchange = async (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  try {
    await streamFile(file);
  } catch (err) {
    setStatus(`lỗi file: ${err.message}`, "err");
    log("error", err.message);
  }
};

ui.pauseBtn.onclick = () => {
  const f = state.file;
  if (!f) return;
  if (f.paused) {
    // Bấm tiếp tục thì bỏ luôn chuỗi nghe lại đang dở.
    state.review = null;
    stopPlayback();
    if (state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ action: "cancel_tts" }));
    }
    resumePlayback();
  } else {
    pausePlayback("bạn đã tạm dừng");
  }
};

function setTtsMode(mode) {
  if (state.ws?.readyState !== WebSocket.OPEN) return;
  state.ws.send(JSON.stringify({ action: "set_tts_mode", mode }));
  log("file", `chế độ TTS: ${mode}`);
}

ui.autoPause.onchange = () => {
  setTtsMode(ui.autoPause.checked ? "manual" : "auto");
  if (!ui.autoPause.checked) {
    state.review = null;
    ui.reviewSteps.hidden = true;
    if (state.file?.paused) resumePlayback();
  }
};

ui.toggle.onclick = () => (state.running ? stop() : start());
ui.stopTts.onclick = () => {
  stopPlayback();
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action: "cancel_tts" }));
  }
};
