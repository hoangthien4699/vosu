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
  close() { return Promise.resolve(); }
  get destination() { return {}; }
}
globalThis.AudioContext = Ctx;
Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });

/* ---- nạp app.js thật ---- */
let src = fs.readFileSync(path.join(ROOT, "app.js"), "utf8");
src += "\nglobalThis.__app = { state, onEvent };\n";
const tmp = path.join(process.env.TMPDIR || "/tmp", "app_smoke.mjs");
fs.writeFileSync(tmp, src);
await import(tmp);
const { state, onEvent } = globalThis.__app;

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

console.log(JSON.stringify({ problems }));
process.exit(0);
