"""임의 CTC 모델을 받아 주어진 wav들을 전사. 대체 모델 A/B용."""
import sys

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

name = sys.argv[1]
wavs = sys.argv[2:] or ["recordings/001_국물.wav"]

proc = Wav2Vec2Processor.from_pretrained(name)
model = Wav2Vec2ForCTC.from_pretrained(name)
model.eval()


def decode(y):
    inp = proc(np.asarray(y, np.float32), sampling_rate=16000,
               return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inp.input_values).logits
    return proc.batch_decode(torch.argmax(logits, dim=-1))[0]


print(f"=== {name} ===")
for w in wavs:
    y = librosa.load(w, sr=16000, mono=True)[0]
    yt, _ = librosa.effects.trim(y, top_db=30)
    yt = yt / (np.abs(yt).max() + 1e-9) * 0.95
    print(f"{w}: 원본=「{decode(y)}」  트림+정규화=「{decode(yt)}」")
