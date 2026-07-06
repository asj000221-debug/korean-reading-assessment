"""목록(긴 발화) 읽기 파이프라인.

여러 단어를 이어서 읽은 녹음 하나를 인식하면 모델이 짧은단어 사각지대를 피해
잘 잡는다(연결발화 1.5s+). 그걸 각 단어에 정렬해서 단어별로 채점한다.
"""

import json
import os

from align import diff_errors_by_word
from g2p_expected import expected_prons
from scoring import WordScore, format_report, score_from_errors

HERE = os.path.dirname(os.path.abspath(__file__))


def load_items(path=None):
    path = path or os.path.join(HERE, "items.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", []), set(data.get("real_words", []))


def score_list(items, actual_text):
    """items(순서 list of {word,type}) + 연결발화 인식 → 단어별 WordScore 리스트."""
    prons = [expected_prons(it["word"], it["type"])[0] for it in items]
    per_word_errors = diff_errors_by_word(prons, actual_text)
    scores = []
    for it, pron, errs in zip(items, prons, per_word_errors):
        # 목록 모드에선 단어별 actual 표면형 복원이 어려워 transparent 실단어판정은 보류(관대)
        scores.append(
            score_from_errors(it["word"], it["type"], pron, errs, became_real_word=False)
        )
    return scores, prons


def run_wav(wav_path, items=None):
    if items is None:
        items, _ = load_items()
    from asr import transcribe

    actual = transcribe(wav_path).strip()
    print(f"인식(전체): 「{actual}」")
    print(f"기대(전체): 「{' '.join(expected_prons(it['word'], it['type'])[0] for it in items)}」\n")
    scores, _ = score_list(items, actual)
    print(format_report(scores))
    return scores, actual


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--dry", help="연결발화 인식 결과를 직접 입력해 채점 로직 점검")
    args = ap.parse_args()
    items, _ = load_items()
    if args.dry:
        print(f"인식(전체): 「{args.dry}」")
        scores, _ = score_list(items, args.dry)
        print(format_report(scores))
    elif args.wav:
        run_wav(args.wav, items)
    else:
        print("--wav <파일> 또는 --dry <인식문자열> 필요")
