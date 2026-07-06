"""깨끗한 데이터셋 음성을 짧게 잘라 '단어 단위' 인식이 되는지 검증.
브라우저 오디오 처리 vs 짧은단어 난이도, 원인 가리기."""
import numpy as np
import torch
from datasets import load_dataset

from asr import _load

processor, model = _load()


def decode(y, sr=16000):
    if sr != 16000:
        import librosa
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=16000)
    inp = processor(np.asarray(y, np.float32), sampling_rate=16000,
                    return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inp.input_values).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]


ds = load_dataset("Bingsu/zeroth-korean", split="test", streaming=True)
ex = next(iter(ds))
y = np.asarray(ex["audio"]["array"], np.float32)
sr = ex["audio"]["sampling_rate"]
print(f"전체({len(y)/sr:.1f}s): {decode(y, sr)}")
print(f"정답: {ex['text']}\n")

# 앞에서 0.6s, 0.8s, 1.0s, 1.5s 만 잘라서(=단어 1~2개 분량) 인식
for sec in (0.6, 0.8, 1.0, 1.5, 2.0):
    seg = y[: int(sec * sr)]
    print(f"앞 {sec}s -> 「{decode(seg, sr)}」")
