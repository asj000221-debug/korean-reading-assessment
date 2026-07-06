"""트랙2 — 음소 인식 백엔드 (slplab CTC).

한글이 아니라 음소(로마자)를 출력한다. 읽기평가의 정밀도 경로(브리프 §8).
오독을 음소 단위로 더 충실히 받아적고, 1음절·짧은 단어도 kresnik보다 잘 잡는다.
"""

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

from asr import TARGET_SR, load_audio, preprocess

MODEL_NAME = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"


@lru_cache(maxsize=1)
def _load():
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME).eval()
    return proc, model


# 긴 오디오는 통째로 넣지 않는다: wav2vec2 self-attention이 프레임수²라
# 60초(≈3000프레임)면 메모리·시간이 급증(느린 PC에선 OOM/연결끊김 → 결과 공백).
# 무음 기준으로 잘라 이 길이 이하 청크로 인식 후 이어붙인다(낱말목록/단락처럼
# 긴 연결발화의 핵심 경로).
_CHUNK_SEC = 18.0       # 청크 최대 길이(초)
_LONG_SEC = 22.0        # 이 이상이면 청크 분할 발동


def _decode(speech, proc, model):
    inputs = proc(speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    return proc.batch_decode(torch.argmax(logits, dim=-1))[0].strip()


def _chunk_bounds(speech):
    """무음 기준 비-무음 구간을 _CHUNK_SEC 이하 청크로 묶어 (start,end) 리스트 반환.

    발화 중간(음소 내부)에서 자르지 않도록 무음 경계에서만 분할한다. 한 발화가
    청크 한도를 넘으면 그 구간은 한도로 강제 분할(드묾)."""
    import librosa
    cap = int(_CHUNK_SEC * TARGET_SR)
    ints = librosa.effects.split(speech, top_db=30)  # [[s,e],...] 비-무음 구간
    if len(ints) == 0:
        return [(0, len(speech))]
    chunks, cs, ce = [], int(ints[0][0]), int(ints[0][1])
    for s, e in ints[1:]:
        s, e = int(s), int(e)
        if e - cs <= cap:            # 현재 청크에 이어붙여도 한도 이내
            ce = e
        else:
            chunks.append((cs, ce))
            cs, ce = s, e
    chunks.append((cs, ce))
    # 단일 구간이 한도 초과 시 강제 분할
    out = []
    for cs, ce in chunks:
        if ce - cs <= cap:
            out.append((cs, ce))
        else:
            for p in range(cs, ce, cap):
                out.append((p, min(p + cap, ce)))
    return out


def transcribe_phones(wav_path, do_preprocess=True):
    """16kHz mono wav → 음소열 문자열(공백 구분, 예 'G U NG M U L').

    긴 녹음은 무음 경계 청크로 나눠 인식 후 이어붙인다(메모리·시간 상한)."""
    proc, model = _load()
    speech = load_audio(wav_path)
    if do_preprocess:
        speech = preprocess(speech)

    if len(speech) <= _LONG_SEC * TARGET_SR:
        return _decode(speech, proc, model)

    parts = []
    for cs, ce in _chunk_bounds(speech):
        seg = speech[cs:ce]
        if len(seg) < int(0.05 * TARGET_SR):   # 너무 짧은 조각은 건너뜀
            continue
        txt = _decode(seg, proc, model)
        if txt:
            parts.append(txt)
    return " ".join(parts).strip()


if __name__ == "__main__":
    import sys
    print(transcribe_phones(sys.argv[1]))
