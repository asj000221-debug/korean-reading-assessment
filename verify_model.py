"""모델이 '깨끗한 한국어'에서 제대로 동작하는지 검증.
zeroth-korean 테스트 샘플 1개를 스트리밍으로 받아 전사."""
import numpy as np
import torch

from asr import _load

processor, model = _load()


def decode(y, sr=16000):
    if sr != 16000:
        import librosa
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=16000)
    inp = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inp.input_values).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]


from datasets import load_dataset

ds = load_dataset("Bingsu/zeroth-korean", split="test", streaming=True)
for i, ex in enumerate(ds):
    if i >= 3:
        break
    a = ex["audio"]
    y, sr = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
    pred = decode(y, sr)
    print(f"\n[{i}] dur={len(y)/sr:.1f}s sr={sr}")
    print(f"  정답: {ex['text']}")
    print(f"  인식: {pred}")
