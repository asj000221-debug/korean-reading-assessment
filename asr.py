"""wav2vec2 CTC 추론.

Whisper류는 못 쓴다 — 디코더 LM이 오독을 자동 보정(self-correction)해버려서
읽기평가가 안 된다. CTC(wav2vec2)는 외부 LM 없이 argmax라 들린 대로 뱉는다.
오디오는 반드시 16kHz mono float. 안 맞으면 인식이 통째로 망가진다.
"""

from functools import lru_cache

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# v0 기본 모델. 한국어 음절(한글) 출력 → 이후 자모 분해해서 사용(align.py).
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
TARGET_SR = 16000


@lru_cache(maxsize=1)
def _load():
    """모델·프로세서는 한 번만 로드."""
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def load_audio(wav_path):
    """16kHz mono float32로 강제 변환해 로드."""
    speech, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    return speech


def preprocess(speech, top_db=30):
    """단어 단위 녹음 보정: 앞뒤 무음 트림 + 피크 정규화.

    문장이 아닌 '단어 1개'는 앞뒤 무음이 길고 음량이 약한 경우가 많아
    인식이 깨진다. 트림으로 발화 구간만 남기고, 약한 녹음을 정규화한다.
    """
    import numpy as np

    trimmed, _ = librosa.effects.trim(speech, top_db=top_db)
    if len(trimmed) < TARGET_SR * 0.1:  # 트림이 과하게 다 깎으면 원본 유지
        trimmed = speech
    peak = float(np.abs(trimmed).max()) if len(trimmed) else 0.0
    if peak > 1e-4:
        trimmed = trimmed / peak * 0.95
    # 앞뒤 0.1s 무음 패딩(경계 음소 보호)
    pad = np.zeros(int(TARGET_SR * 0.1), dtype=trimmed.dtype)
    return np.concatenate([pad, trimmed, pad])


def audio_quality(speech):
    """녹음 품질 지표(webapp 품질 게이트용)."""
    import numpy as np

    peak = float(np.abs(speech).max()) if len(speech) else 0.0
    clip = float(np.mean(np.abs(speech) > 0.98)) if len(speech) else 0.0
    voiced, _ = librosa.effects.trim(speech, top_db=30)
    voiced_sec = len(voiced) / TARGET_SR
    return {
        "peak": peak,
        "clip": clip,
        "voiced_sec": voiced_sec,
        "total_sec": len(speech) / TARGET_SR,
    }


def transcribe(wav_path):
    """16kHz mono .wav → 인식 텍스트(한글).

    CTC argmax 디코딩이므로 LM 보정 없이 들린 대로 출력한다.
    단어 녹음 보정(트림+정규화)을 적용한다.
    """
    processor, model = _load()
    speech = preprocess(load_audio(wav_path))
    inputs = processor(
        speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python asr.py <wav_path>")
        raise SystemExit(1)
    print(transcribe(sys.argv[1]))
