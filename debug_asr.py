"""저장된 녹음 파일로 ASR을 다각도 디버깅."""
import sys

import librosa
import numpy as np
import torch

from asr import _load

path = sys.argv[1] if len(sys.argv) > 1 else "recordings/001_국물.wav"

# 1) 원본 진단
y_raw, sr_raw = librosa.load(path, sr=None, mono=True)
print(f"원본: {len(y_raw)/sr_raw:.2f}s, {sr_raw}Hz, peak={np.abs(y_raw).max():.3f}, "
      f"rms={np.sqrt(np.mean(y_raw**2)):.4f}")

processor, model = _load()


def decode(y):
    inp = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inp.input_values).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]


# 2) 표준 16k 변환 후
y16 = librosa.load(path, sr=16000, mono=True)[0]
print(f"\n[A] 표준 16k         -> 「{decode(y16)}」")

# 3) 침묵 트리밍(top_db=30, 20)
for top_db in (30, 20, 40):
    yt, _ = librosa.effects.trim(y16, top_db=top_db)
    print(f"[B] trim top_db={top_db:<3} ({len(yt)/16000:.2f}s) -> 「{decode(yt)}」")

# 4) 피크 정규화
ynorm = y16 / (np.abs(y16).max() + 1e-9) * 0.95
print(f"\n[C] peak normalize    -> 「{decode(ynorm)}」")

# 5) 트림 + 정규화
yt, _ = librosa.effects.trim(y16, top_db=25)
ytn = yt / (np.abs(yt).max() + 1e-9) * 0.95
print(f"[D] trim25+norm ({len(yt)/16000:.2f}s) -> 「{decode(ytn)}」")

# 6) 앞뒤 0.1s 패딩 추가(트림 후)
ypad = np.concatenate([np.zeros(1600), ytn, np.zeros(1600)]).astype(np.float32)
print(f"[E] trim+norm+pad     -> 「{decode(ypad)}」")
