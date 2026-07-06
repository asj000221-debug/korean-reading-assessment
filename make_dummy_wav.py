"""STEP 1 검증용 더미 16kHz mono wav 생성.

실제 음성이 아니므로 인식 결과는 무의미하다. 목적은 오직
transcribe() 파이프라인(로드→전처리→추론→디코딩)이 끝까지 도는지 확인.
실제 self-correction 검증(§2-3)은 사람이 녹음한 오독 샘플로 별도 수행.
"""

import numpy as np
import soundfile as sf

SR = 16000


def make_dummy(path="dummy_16k.wav", seconds=1.5):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    # 약한 잡음 + 220Hz 톤. 클리핑 방지로 진폭 작게.
    rng = np.random.default_rng(0)
    sig = 0.05 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(t.shape)
    sf.write(path, sig.astype(np.float32), SR)
    print(f"wrote {path}  sr={SR} mono frames={len(sig)}")
    return path


if __name__ == "__main__":
    make_dummy()
