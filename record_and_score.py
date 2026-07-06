"""직접 발음해서 채점해보는 대화형 도구.

마이크 접근은 '사용자 본인의 터미널 세션'에서만 된다(에이전트 프로세스는 오디오 세션 없음).
따라서 이 스크립트는 사용자가 직접 실행한다:

    python record_and_score.py                 # items.json 전체를 순서대로
    python record_and_score.py --word 책상 --type transparent
    python record_and_score.py --list-devices  # 마이크 장치 번호 확인
    python record_and_score.py --device 14     # 특정 마이크 지정

흐름: 단어 표시 → [Enter] → 카운트다운 → 녹음 → ASR → 채점 → 결과 출력 → 다음.
오독이 '보정되지 않고' 잡히는지(§2-3) 직접 확인하는 용도.
"""

import argparse
import json
import os

import numpy as np
import sounddevice as sd
import soundfile as sf

from g2p_expected import expected_prons
from scoring import score_word

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))


def pick_input_device(device):
    """입력 장치 결정. 지정 없으면 기본 입력장치, 그래도 안되면 첫 입력장치."""
    if device is not None:
        return device
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            return default_in
    except Exception:
        pass
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            return i
    raise RuntimeError("입력(마이크) 장치를 찾지 못했습니다. --list-devices 로 확인하세요.")


def record(seconds, device):
    """카운트다운 후 seconds 초 녹음 → 16kHz mono float32 numpy."""
    print("   준비... ", end="", flush=True)
    for n in (3, 2, 1):
        print(f"{n} ", end="", flush=True)
        sd.sleep(700)
    print("🎤 녹음 중! 지금 읽으세요...", flush=True)
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    audio = audio.reshape(-1)
    peak = float(np.abs(audio).max())
    print(f"   (녹음 끝, 최대진폭 {peak:.3f})")
    if peak < 0.01:
        print("   ⚠ 소리가 거의 없습니다. 마이크/장치 번호를 확인하세요.")
    return audio


def transcribe_array(audio):
    """numpy 오디오 → ASR 텍스트 (asr 모델 재사용)."""
    import torch

    from asr import _load
    processor, model = _load()
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()


def load_items():
    with open(os.path.join(HERE, "items.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", []), set(data.get("real_words", []))


def run_one(word, item_type, real_words, device, seconds):
    exp = expected_prons(word, item_type)[0]
    print(f"\n▶ 읽을 단어:  「{word}」   (유형 {item_type}, 기대발음 [{exp}])")
    input("   [Enter] 누르면 녹음 시작...")
    audio = record(seconds, device)
    actual = transcribe_array(audio)
    print(f"   인식 결과:  「{actual}」")
    s = score_word(word, item_type, exp, actual, real_words=real_words)
    mark = "✅ 정답" if s.correct else "❌ 오류"
    print(f"   채점:  {mark}  (정확도 {s.accuracy:.0%})")
    for e in s.errors:
        ej = "".join(t.char for t in e.exp) or "-"
        aj = "".join(t.char for t in e.act) or "-"
        print(f"      - {e.kind} [{e.tag}] {ej}->{aj} (감점 {e.weight})")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word")
    ap.add_argument("--type", choices=["nonword", "phonrule", "transparent"])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"{i:3}  {d['name']}  (in {d['max_input_channels']}ch)")
        return

    device = pick_input_device(args.device)
    print(f"사용 마이크 장치: {device} ({sd.query_devices(device)['name']})")
    print("ASR 모델 로딩 중...")
    from asr import _load
    _load()

    items, real_words = load_items()
    if args.word:
        run_one(args.word, args.type or "transparent", real_words, device, args.seconds)
        return

    print(f"\nitems.json {len(items)}개 문항을 순서대로 진행합니다. (Ctrl+C로 중단)")
    results = []
    for it in items:
        results.append(run_one(it["word"], it["type"], real_words, device, args.seconds))
    n_ok = sum(1 for s in results if s.correct)
    print(f"\n=== 전체 {n_ok}/{len(results)} 정답 ===")


if __name__ == "__main__":
    main()
