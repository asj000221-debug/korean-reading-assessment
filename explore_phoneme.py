"""트랙2 사전조사: 음소 모델의 출력 형식 파악."""
import sys

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCTC

name = sys.argv[1] if len(sys.argv) > 1 else "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"
wavs = sys.argv[2:] or ["recordings/list_001.wav", "recordings/001_국물.wav"]

print(f"=== {name} ===")
proc = AutoProcessor.from_pretrained(name)
model = AutoModelForCTC.from_pretrained(name).eval()

# vocab 일부 출력
try:
    tok = proc.tokenizer
    vocab = tok.get_vocab()
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    print(f"vocab size={len(vocab)} 예시:", [k for k, _ in items[:40]])
except Exception as e:
    print("vocab dump 실패:", e)


def decode(y):
    inp = proc(np.asarray(y, np.float32), sampling_rate=16000,
               return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inp.input_values).logits
    ids = torch.argmax(logits, dim=-1)
    return proc.batch_decode(ids)[0]


for w in wavs:
    try:
        y = librosa.load(w, sr=16000, mono=True)[0]
        print(f"\n{w} -> 「{decode(y)}」")
    except Exception as e:
        print(f"{w}: {e}")
