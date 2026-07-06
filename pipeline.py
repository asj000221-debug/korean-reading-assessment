"""전체 파이프라인 — wav + 문항 → 점수 (§5).

흐름:
  items.json 로드 (단어·유형)
   → g2p_expected 로 기대 발음 생성
   → asr.transcribe 로 실제 발화 인식 (또는 dry-run에서 actual 주입)
   → align.diff_errors 로 오류 추출
   → scoring 으로 유형별 채점
   → 단어별/전체 정확도 리포트 출력

사용법:
  실음성:  python pipeline.py --items items.json --wav-dir ./wavs
           (wavs/<word>.wav 를 각 문항에 매칭, 또는 items의 "wav" 필드 사용)
  dry-run: python pipeline.py --items items.json --dry-run responses.json
           (responses.json: {"단어": "인식결과", ...} — ASR 없이 채점 로직만 점검)
"""

import argparse
import json
import os

from g2p_expected import expected_prons
from scoring import format_report, score_word


def load_items(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", []), set(data.get("real_words", []))


def resolve_actual(item, wav_dir, dry_responses):
    """문항의 실제 발화 텍스트(actual_text)를 얻는다."""
    word = item["word"]
    if dry_responses is not None:
        return dry_responses.get(word)
    # 실음성 경로: item["wav"] 우선, 없으면 wav_dir/<word>.wav
    wav = item.get("wav")
    if wav is None and wav_dir:
        cand = os.path.join(wav_dir, f"{word}.wav")
        wav = cand if os.path.exists(cand) else None
    if wav is None:
        return None
    from asr import transcribe  # 무거운 import는 실제 필요할 때만

    return transcribe(wav)


def run(items_path, wav_dir=None, dry_path=None):
    items, real_words = load_items(items_path)
    dry_responses = None
    if dry_path:
        with open(dry_path, encoding="utf-8") as f:
            dry_responses = json.load(f)

    scores = []
    for item in items:
        word, item_type = item["word"], item["type"]
        exp = item.get("expected_pron") or expected_prons(word, item_type)[0]
        actual = resolve_actual(item, wav_dir, dry_responses)
        if actual is None:
            print(f"[skip] {word}: 발화(actual) 없음 — wav 또는 dry 응답 누락")
            continue
        scores.append(
            score_word(word, item_type, exp, actual, real_words=real_words)
        )

    print(format_report(scores))
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="items.json")
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--dry-run", dest="dry", default=None,
                    help="JSON {단어: 인식결과} — ASR 없이 채점 로직 점검")
    args = ap.parse_args()
    run(args.items, wav_dir=args.wav_dir, dry_path=args.dry)
