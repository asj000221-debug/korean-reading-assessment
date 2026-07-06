"""음소 모델 기반 목록 읽기 채점.

items → 기대발음(한글) → 기대 음소열 → 실제 음소열(phoneme_asr) →
단어별 DP 음소정렬 → 채점.
"""

import json
import os

from align import _key_char, align_tokens_dp, diff_tokens
from g2p_expected import expected_prons
from phoneme_map import phones_from_model, phones_to_hangul, to_phone_tokens
from scoring import score_from_errors

HERE = os.path.dirname(os.path.abspath(__file__))


def load_items(path=None):
    path = path or os.path.join(HERE, "items.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", []), set(data.get("real_words", []))


def score_list_phoneme(items, actual_phone_text):
    """items(순서) + 모델 음소열 → (단어별 WordScore, 들린 한글 리스트).

    각 단어에 배정된 음소 구간을 한글로 복원해 '들린 대로'를 보여준다.
    """
    prons = [expected_prons(it["word"], it["type"])[0] for it in items]
    exp_word_tokens = [to_phone_tokens(p) for p in prons]
    act_tokens = phones_from_model(actual_phone_text)
    segs = align_tokens_dp(exp_word_tokens, act_tokens, _key_char)

    scores, heard = [], []
    for it, pron, toks, seg in zip(items, prons, exp_word_tokens, segs):
        errs = diff_tokens(toks, seg, _key_char)
        scores.append(
            score_from_errors(it["word"], it["type"], pron, errs,
                              became_real_word=False, n_units=len(toks))
        )
        heard.append(phones_to_hangul([t.char for t in seg]))
    return scores, heard


GOP_OK = -2.0  # 기대 음소 GOP가 이 이상이면 '발음했다'로 인정(ASR argmax 오인식 구제)


def apply_gop(items, wav_path, scores, gop_thresh=GOP_OK):
    """GOP로 판정 보정: 각 단어의 최저 기대음소 GOP를 보고
    verdict('correct'|'uncertain') + 최저 GOP를 반환.

    - 자유인식이 정답이거나, 최저 GOP ≥ 임계값이면 correct(틀림 구제).
    - 그 외(자유인식 오류 + 최저 GOP 낮음)는 uncertain(검수 필요).
    """
    from gop import gop_by_word

    etoks = [to_phone_tokens(expected_prons(it["word"], it["type"])[0]) for it in items]
    per_word_gop, _ = gop_by_word(wav_path, etoks)
    verdicts, min_gops = [], []
    for s, gl in zip(scores, per_word_gop):
        gs = [g for _, g in gl if g is not None]
        mg = min(gs) if gs else None
        min_gops.append(mg)
        if s.correct or (mg is not None and mg >= gop_thresh):
            verdicts.append("correct")
        else:
            verdicts.append("uncertain")
    return verdicts, min_gops


def format_report_hangul(scores, heard):
    """사람이 읽을 수 있는 한글 리포트: 목표 → 들린대로, 정/오."""
    lines = []
    n_ok = sum(1 for s in scores if s.correct)
    for s, h in zip(scores, heard):
        mark = "✅" if s.correct else "❌"
        h = h or "(못 들음)"
        lines.append(f"{mark} {s.word} ({s.item_type})  →  들린대로: {h}   정확도 {s.accuracy:.0%}")
    lines.append(f"\n=== {n_ok}/{len(scores)} 정답 ===")
    return "\n".join(lines)


def run_wav(wav_path, items=None):
    if items is None:
        items, _ = load_items()
    from phoneme_asr import transcribe_phones

    actual = transcribe_phones(wav_path)
    scores, heard = score_list_phoneme(items, actual)
    print(f"들린 전체(한글): 「{phones_to_hangul(actual)}」\n")
    print(format_report_hangul(scores, heard))
    return scores, actual


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--dry", help="모델 음소열을 직접 입력해 채점 로직 점검")
    args = ap.parse_args()
    items, _ = load_items()
    if args.dry:
        scores, heard = score_list_phoneme(items, args.dry)
        print(f"들린 전체(한글): 「{phones_to_hangul(args.dry)}」\n")
        print(format_report_hangul(scores, heard))
    elif args.wav:
        run_wav(args.wav, items)
    else:
        print("--wav <파일> 또는 --dry <음소열> 필요")
