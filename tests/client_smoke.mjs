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

/* ---- DOM giả ----
 *
 * Đủ THẬT để chạy được code dựng lịch sử bản dịch: có con, có querySelector,
 * có classList thật. Bản giả sơ sài hơn sẽ báo lỗi giả ("querySelector is not
 * a function") và cám dỗ sửa code cho vừa bản giả — ngược đời.                */
const nodes = new Map();

function mk(id, tag = "div") {
  const el = {
    id, tag, value: "", checked: id === "autoPause",
    disabled: false, hidden: false, dataset: {}, style: {},
    _children: [], _text: "", _classes: new Set(),
    classList: {
      add(...c) { c.forEach((x) => el._classes.add(x)); },
      remove(...c) { c.forEach((x) => el._classes.delete(x)); },
      toggle(c, on) { if (on === undefined) { el._classes.has(c) ? el._classes.delete(c) : el._classes.add(c); } else if (on) { el._classes.add(c); } else { el._classes.delete(c); } },
      contains(c) { return el._classes.has(c); },
    },
    get className() { return [...el._classes].join(" "); },
    set className(v) { el._classes = new Set(String(v).split(/\s+/).filter(Boolean)); },
    get textContent() { return el._text; },
    set textContent(v) { el._text = String(v); el._children = []; },
    get children() { return el._children; },
    get childElementCount() { return el._children.length; },
    get firstChild() { return el._children[0] ?? null; },
    get lastChild() { return el._children[el._children.length - 1] ?? null; },
    get lastElementChild() { return el._children[el._children.length - 1] ?? null; },
    get scrollHeight() { return el._children.length * 40; },
    get clientHeight() { return 400; },
    scrollTop: 0,
    appendChild(c) { el._children.push(c); return c; },
    prepend(c) { el._children.unshift(c); return c; },
    removeChild(c) { el._children = el._children.filter((x) => x !== c); return c; },
    remove() {},
    addEventListener() {}, click() {},
    querySelector(sel) {
      // Đủ cho hai dạng code thật dùng: `.msg[data-utt="..."]` và `.txt`.
      const attr = /^\.(\S+?)\[data-utt="(.*)"\]$/.exec(sel);
      const plain = /^\.([\w-]+)$/.exec(sel);
      const khop = (c) => {
        if (attr) return c._classes?.has(attr[1]) && c.dataset?.utt === attr[2];
        if (plain) return c._classes?.has(plain[1]);
        return false;
      };
      const walk = (n) => {
        for (const c of n._children || []) {
          if (khop(c)) return c;
          const deep = walk(c);
          if (deep) return deep;
        }
        return null;
      };
      return walk(el);
    },
  };
  return el;
}
globalThis.document = {
  getElementById: (id) => (nodes.has(id) ? nodes.get(id) : nodes.set(id, mk(id)).get(id)),
  createElement: (tag) => mk("", tag), addEventListener() {},
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
src += "\nglobalThis.__app = { state, onEvent, streamFile, start, stop };\n";
const tmp = path.join(process.env.TMPDIR || "/tmp", "app_smoke.mjs");
fs.writeFileSync(tmp, src);
await import(tmp);
const { state, onEvent, streamFile, start, stop } = globalThis.__app;

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

/* ---- lịch sử bản dịch: câu mới KHÔNG được xóa câu cũ ---- */
/* Trước đây bản dịch là MỘT dòng bị ghi đè mỗi câu, nên khi người ta nói nhanh
 * thì câu trước biến mất trước khi đọc kịp. */
const chat = document.getElementById("chat");
chat.textContent = "";
const BA_CAU = [
  ["utt_101", "First thing they said.", "Câu thứ nhất."],
  ["utt_102", "Second thing.", "Câu thứ hai."],
  ["utt_103", "Third thing.", "Câu thứ ba."],
];
for (const [id, src, dich] of BA_CAU) {
  onEvent({ session_id: "s", utterance_id: id, sequence: 1, type: "copilot_started",
            timestamp: "", data: { source_text: src, language: "en", direction: "to_user" } });
  onEvent({ session_id: "s", utterance_id: id, sequence: 2, type: "translation_delta",
            timestamp: "", data: { text: dich, full: dich, direction: "to_user", language: "vi" } });
  onEvent({ session_id: "s", utterance_id: id, sequence: 3, type: "copilot_done",
            timestamp: "", data: { ttft_ms: 1, total_ms: 2, tokens: 3, truncated: false } });
}
if (chat.childElementCount !== 3) {
  problems.push(`đáng lẽ giữ 3 câu trong lịch sử, thực tế ${chat.childElementCount}`);
} else {
  const dau = chat.children[0].querySelector(".txt").textContent;
  const cuoi = chat.children[2].querySelector(".txt").textContent;
  if (dau !== "Câu thứ nhất.") {
    problems.push("câu ĐẦU bị mất hoặc ghi đè: " + dau);
  }
  if (cuoi !== "Câu thứ ba.") problems.push("câu mới nhất không nằm dưới cùng");
  if (chat.children[2]._classes.has("streaming")) {
    problems.push("câu đã dịch xong vẫn còn đánh dấu đang chạy");
  }
}

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

/* ---- nghe âm thanh thiết bị (YouTube đang phát trên máy) ---- */
class FakeTrack {
  constructor(kind) { this.kind = kind; this.stopped = false; this._l = {}; }
  stop() { this.stopped = true; }
  addEventListener(t, fn) { (this._l[t] ??= []).push(fn); }
}
function fakeStream(withAudio) {
  const tracks = [new FakeTrack("video")];
  if (withAudio) tracks.push(new FakeTrack("audio"));
  return {
    _tracks: tracks,
    getTracks() { return this._tracks; },
    getVideoTracks() { return this._tracks.filter((t) => t.kind === "video"); },
    getAudioTracks() { return this._tracks.filter((t) => t.kind === "audio"); },
    removeTrack(t) { this._tracks = this._tracks.filter((x) => x !== t); },
  };
}
let displayCalls = 0, shared = null, videoTrack = null;
Object.defineProperty(globalThis, "navigator", {
  value: { mediaDevices: {
    getDisplayMedia: async () => {
      displayCalls += 1;
      shared = fakeStream(true);
      videoTrack = shared.getVideoTracks()[0];
      return shared;
    },
    getUserMedia: async () => { problems.push("nguồn 'device' lại đi gọi getUserMedia"); return fakeStream(true); },
  } },
  configurable: true,
});
// AudioWorklet không có trong Node — chỉ cần chạy tới đó là đủ chứng minh
// đường đi đúng nguồn; phần sau là API trình duyệt, không phải logic của ta.
Ctx.prototype.createMediaStreamSource = function () { return { connect() {} }; };
Ctx.prototype.createGain = function () { return { gain: {}, connect() {} }; };
Object.defineProperty(Ctx.prototype, "audioWorklet", {
  value: { addModule: async () => { throw new Error("__đã tới AudioWorklet__"); } },
  configurable: true,
});
globalThis.URL.createObjectURL = () => "blob:x";
globalThis.URL.revokeObjectURL = () => {};
globalThis.Blob = class { constructor() {} };

await start("device");
if (displayCalls !== 1) problems.push(`đáng lẽ gọi getDisplayMedia 1 lần, thực tế ${displayCalls}`);
if (shared && shared.getVideoTracks().length) {
  problems.push("track video không được gỡ khỏi stream");
}
if (videoTrack && !videoTrack.stopped) {
  problems.push("track video chưa stop() — camera/quay màn hình vẫn chạy vô ích");
}

/* nguồn không kèm audio -> phải báo lỗi rõ, không im lặng chạy tiếp */
globalThis.navigator.mediaDevices.getDisplayMedia = async () => fakeStream(false);
stop();
await start("device");
const status = document.getElementById("status").textContent;
// Chuỗi phải ĐẶC TRƯNG cho đúng lỗi này. Bắt mỗi "âm thanh" thì mọi lỗi khác
// cũng khớp — assertion đó không kiểm được gì (đã mắc một lần).
if (!/không kèm âm thanh/i.test(status)) {
  problems.push("nguồn không kèm audio mà không báo rõ, chỉ hiện: " + status);
}
if (state.running) problems.push("nguồn không kèm audio mà vẫn coi như đang chạy");

console.log(JSON.stringify({ problems }));
process.exit(0);
