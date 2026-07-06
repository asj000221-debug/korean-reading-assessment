"""time-stretch(음정유지 속도감속)로 짧은 단어 인식이 살아나는지 검증."""
import sys

import librosa
import numpy as np
import torch

from asr import _load, load_audio, preprocess

processor, model = _load()


def decode(y):
    inp = processor(np.asarray(y, np.float32), sampling_rate=16000,
                    return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inp.input_values).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]


path = sys.argv[1] if len(sys.argv) > 1 else "recordings/001_국물.wav"
y = load_audio(path)
yt, _ = librosa.effects.trim(y, top_db=30)
yt = yt / (np.abs(yt).max() + 1e-9) * 0.95
print(f"트림 후 발화: {len(yt)/16000:.2f}s")
print(f"[원본 속도]        -> 「{decode(yt)}」")

# rate<1 이면 느려짐(길어짐). 0.5 => 2배 길이
for rate in (0.8, 0.6, 0.5, 0.4):
    ys = librosa.effects.time_stretch(yt, rate=rate)
    pad = np.zeros(1600, np.float32)
    ys = np.concatenate([pad, ys, pad])
    print(f"[rate={rate} → {len(ys)/16000:.2f}s] -> 「{decode(ys)}」")
