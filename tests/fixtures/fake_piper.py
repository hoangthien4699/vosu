#!/usr/bin/env python3
"""Giả lập Piper: đọc 1 dòng text từ stdin, phát PCM16 ra stdout dần dần.

`--output-raw` của Piper thật cũng ghi PCM thô ra stdout, nên hợp đồng I/O
giống hệt. Tham số VOSU_FAKE_PIPER_DELAY dùng để mô phỏng giọng đọc dài,
cho phép test Barge-in cắt ngang giữa chừng.
"""
import os
import sys
import time


def main() -> None:
    line = sys.stdin.readline()
    if not line.strip():
        return
    delay = float(os.environ.get("VOSU_FAKE_PIPER_DELAY", "0.01"))
    total = int(os.environ.get("VOSU_FAKE_PIPER_CHUNKS", "20"))
    payload = b"\x00\x01" * 1024
    for _ in range(total):
        try:
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            return
        time.sleep(delay)

if __name__ == "__main__":
    main()
