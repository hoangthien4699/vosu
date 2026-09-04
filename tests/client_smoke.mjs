/* Chạy CHÍNH client/web_test/app.js trong Node với DOM + WebAudio giả.
 *
 * Vì sao cần: các test khác chỉ đọc app.js như VĂN BẢN (có case này không, có
 * gọi hàm kia không). Chúng không bắt được lỗi chỉ lộ khi CHẠY. Đã trả giá:
 * `noteBusy(type)` đọc biến `uttId` không tồn tại -> mỗi lượt `tts_done` ném
 * ReferenceError ngay trước dòng resolve() -> chờ đọc xong không bao giờ được
 * giải phóng -> file chỉ phát tiếp nhờ cầu chì 30 giây. Người dùng thấy "im
 * lặng rất lâu" sau mỗi bản dịch. Đo trên client thật: 27.6 giây mỗi câu.
 */
import fs from "node:fs";
import path from "node:path";

const ROOT = process.argv[2];
const problems = [];

/* ---- DOM giả ---- */
const nodes = new Map();
const mk = (id) => ({
  id, textContent: "", className: "", value: "", checked: id === "autoPause",
  disabled: false, hidden: false, children: [], dataset: {}, style: {},
  classList: { add() {}, remove() {}, toggle() {} },
  appendChild() {}, prepend() {}, remove() {},
  addEventListener() {}, click() {},
  get childElementCount() { return 0; }, get lastChild() { return null; },
});
globalThis.document = {
  getElementById: (id) => (nodes.has(id) ? nodes.get(id) : nodes.set(id, mk(id)).get(id)),
  createElement: () => mk("tmp"), addEventListener() {},
};
globalThis.window = globalThis;
globalThis.location = { protocol: "http:", host: "127.0.0.1:8000" };
class Ctx {
  get currentTime() { return performance.now() / 1000; }
  createBuffer(c, n, sr) { return { length: n, sampleRate: sr, duration: n / sr,
    getChannelData: () => new Float32Array(n) }; }
  createBufferSource() { return { buffer: null, connect() {}, start() {}, stop() {} }; }
  async decodeAudioData(buf) {
    const dv = new DataView(buf);
    let pos = 12, off = 0, len = 0, sr = 16000;
    while (pos + 8 <= dv.byteLength) {
      const id = String.fromCharCode(dv.getUint8(pos), dv.getUint8(pos + 1),
                                     dv.getUint8(pos + 2), dv.getUint8(pos + 3));
      const sz = dv.getUint32(pos + 4, true);
      if (id === "fmt ") sr = dv.getUint32(pos + 12, true);
      if (id === "data") { off = pos + 8; len = sz; break; }
      pos += 8 + sz + (sz % 2);
    }
    const n = len / 2, out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = dv.getInt16(off + i * 2, true) / 32768;
    return { numberOfChannels: 1, length: n, sampleRate: sr, duration: n / sr,
             getChannelData: () => out };
  }
  close() { return Promise.resolve(); }
  get destination() { return {}; }
}
globalThis.AudioContext = Ctx;
Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });

/* ---- nạp app.js thật ---- */
let src = fs.readFileSync(path.join(ROOT, "app.js"), "utf8");
src += "\nglobalThis.__app = { state, onEvent, streamFile };\n";
const tmp = path.join(process.env.TMPDIR || "/tmp", "app_smoke.mjs");
fs.writeFileSync(tmp, src);
await import(tmp);
const { state, onEvent, streamFile } = globalThis.__app;

/* ---- payload tối thiểu cho từng loại event backend phát ---- */
const U = "utt_001";
const EVENTS = {
  session_started: { platform: "macos", max_concurrent_sessions: 1, sample_rate: 16000, tts_enabled: true },
  session_ended: { reason: "client_disconnect", utterances: 1 },
  audio_started: { sample_rate: 16000, channels: 1 },
  utterance_endpoint: { start_s: 0, duration_s: 1.0, trigger: "vad_endpoint" },
  utterance_continued: { text: "So what I am", reason: "incomplete", wait_ms: 1200 },
  utterance_dropped: { reason: "backlog", pending: 3 },
  stt_partial: { text: "hi", language: "en", window_s: 1.0 },
  stt_final: { text: "hi", language: "en", duration_s: 1.0, latency_ms: 10, start_s: 0, direction: "to_user" },
  copilot_started: { source_text: "hi", language: "en", direction: "to_user" },
  translation_delta: { text: "xin", full: "xin chào", direction: "to_user", language: "vi" },
  copilot_done: { ttft_ms: 10, total_ms: 20, tokens: 3, truncated: false },
  tts_started: { utterance_field: "translation", text: "xin chào", voice: "vi", sample_rate: 22050 },
  tts_done: { chunks: 1, synthesis_ms: 10, prewarmed: false },
  tts_cancelled: { reason: "barge_in", response_ms: 5, chunks_sent: 1 },
  tts_error: { message: "x", stage: "synthesis" },
  error: { message: "x", code: "y", recoverable: true },
};

for (const [type, data] of Object.entries(EVENTS)) {
  try {
    onEvent({ session_id: "s", utterance_id: U, sequence: 1, type, timestamp: "", data });
  } catch (err) {
    problems.push(`${type}: ${err.message}`);
  }
}

/* ---- kịch bản đã hỏng thật: chờ đọc xong một câu ---- */
let resolved = false;
state.ttsWaiter = () => { resolved = true; };
state.ttsWaitFor = U;
try {
  onEvent({ session_id: "s", utterance_id: U, sequence: 2, type: "tts_done",
            timestamp: "", data: EVENTS.tts_done });
} catch (err) {
  problems.push(`tts_done trong lúc đang chờ: ${err.message}`);
}
if (!resolved) problems.push("tts_done của ĐÚNG câu đang chờ mà không giải phóng chờ");

/* ---- lượt đọc của câu KHÁC thì không được giải phóng ---- */
resolved = false;
state.ttsWaiter = () => { resolved = true; };
state.ttsWaitFor = U;
onEvent({ session_id: "s", utterance_id: "utt_999", sequence: 3, type: "tts_done",
          timestamp: "", data: EVENTS.tts_done });
if (resolved) problems.push("tts_done của câu KHÁC lại giải phóng chờ");

/* ---- chọn file thứ hai mà KHÔNG tải lại trang ---- */
/* Kịch bản thật đã hỏng: `stop()` gọi ws.close() rồi `connect()` mở ngay socket
 * mới. close() chỉ là YÊU CẦU đóng, trả về ngay — server thấy hai kết nối và
 * từ chối cái mới bằng "đã đạt giới hạn 1 session". Phải đóng xong rồi mới mở. */
const sockets = [];
class FakeWS {
  static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
  constructor(url) {
    this.url = url; this.readyState = FakeWS.OPEN; this.binaryType = "";
    this._listeners = {}; this.sent = 0;
    this.openedAt = order++;
    sockets.push(this);
    queueMicrotask(() => this._fire("open", {}));
  }
  addEventListener(type, fn) { (this._listeners[type] ??= []).push(fn); }
  _fire(type, ev) {
    for (const fn of this._listeners[type] || []) fn(ev);
    const direct = this["on" + type];
    if (direct) direct(ev);
  }
  send() { this.sent += 1; }
  close() {
    // Đúng chuẩn WebSocket: close() chỉ chuyển sang CLOSING và trả về ngay.
    // CLOSED chỉ tới cùng sự kiện 'close'. Chính khoảng trễ giữa hai mốc đó
    // là chỗ hỏng — mô hình sai chỗ này thì test không bắt được gì.
    if (this.readyState >= FakeWS.CLOSING) return;
    this.readyState = FakeWS.CLOSING;
    setTimeout(() => {
      this.readyState = FakeWS.CLOSED;
      this.closedAt = order++;
      this._fire("close", {});
    }, 30);
  }
}
let order = 0;
globalThis.WebSocket = FakeWS;

function tinyWav(seconds = 0.3, sr = 16000) {
  const n = Math.floor(sr * seconds), buf = Buffer.alloc(44 + n * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write("WAVE", 8);
  buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28);
  buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(n * 2, 40);
  return { name: "tiny.wav",
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
}

try {
  await streamFile(tinyWav());
  await streamFile(tinyWav());
} catch (err) {
  problems.push(`chọn file thứ hai: ${err.message}`);
}
if (sockets.length < 2) {
  problems.push(`đáng lẽ mở 2 kết nối, thực tế ${sockets.length}`);
} else {
  const [first, second] = sockets;
  if (first.closedAt === undefined) problems.push("kết nối cũ chưa đóng xong");
  else if (first.closedAt > second.openedAt) {
    problems.push("mở kết nối MỚI trước khi đóng kết nối CŨ -> server từ chối");
  }
}

console.log(JSON.stringify({ problems }));
process.exit(0);
