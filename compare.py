"""kresnik vs kids 모델 head-to-head (특수토큰 제거 포함)."""
import re
import sys

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODELS = [
    "kresnik/wav2vec2-large-xlsr-korean",
    "lsnoo/xlsr-53-korean_kids_3_to_5_syl",
]
wavs = sys.argv[1:] or ["recordings/001_국물.wav"]


def clean(s):
    return re.sub(r"<unk>|<s>|</s>|\[PAD\]|\|", "", s).strip()


def load(name):
    p = Wav2Vec2Processor.from_pretrained(name)
    m = Wav2Vec2ForCTC.from_pretrained(name).eval()
    return p, m


def decode(p, m, y):
    inp = p(np.asarray(y, np.float32), sampling_rate=16000,
            return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = m(inp.input_values).logits
    return clean(p.batch_decode(torch.argmax(logits, dim=-1))[0])


loaded = [(n, *load(n)) for n in MODELS]
for w in wavs:
    y = librosa.load(w, sr=16000, mono=True)[0]
    yt, _ = librosa.effects.trim(y, top_db=30)
    yt = yt / (np.abs(yt).max() + 1e-9) * 0.95
    print(f"\n## {w}")
    for n, p, m in loaded:
        print(f"  {n.split('/')[-1]:35} 트림정규화=「{decode(p,m,yt)}」")
