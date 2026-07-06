"""문장/지문 읽기 채점 — 어절 단위 정확도(읽기 유창성의 정확도 부분).

실제 난독증 검사의 '읽기 유창성'은 학년 수준 지문을 소리내어 읽혀
어절(띄어쓰기 단위) 정확도 + 속도를 본다. 여기서는 정확도를 채점한다.

원리: 문장 전체를 G2P → 어절 경계(공백) 보존 → 어절별 기대 음소 →
      음소 모델 인식 → DP 정렬로 어절별 배정 → 채점.
문장 G2P는 어절 넘는 연음/경음화까지 반영(실제 발화와 일치).
"""

import re

from align import _key_char, align_tokens_dp, diff_tokens
from g2p_expected import g2p
from phoneme_map import phones_from_model, phones_to_hangul, to_phone_tokens
from scoring import score_from_errors

_PUNCT = re.compile(r"[.,!?…·\"'“”‘’()]")


def expected_eojeol_prons(sentence):
    """문장 → 어절별 기대 발음(한글). 문장 전체 G2P 후 공백으로 분리."""
    pron = g2p(sentence)
    pron = _PUNCT.sub("", pron)
    return [w for w in pron.split() if w]


def surface_eojeols(sentence):
    """원문 어절(표기) 리스트 — 표시용."""
    s = _PUNCT.sub("", sentence)
    return [w for w in s.split() if w]


def score_sentence(sentence, actual_phone_text, item_type="transparent"):
    """문장 + 모델 음소열 → (어절별 WordScore, 들린 한글 리스트, 표기어절)."""
    eojeols = surface_eojeols(sentence)
    prons = expected_eojeol_prons(sentence)
    # 어절 수와 발음 수가 어긋나면(드묾) 표기 기준으로 맞춤
    n = min(len(eojeols), len(prons))
    eojeols, prons = eojeols[:n], prons[:n]

    exp_tokens = [to_phone_tokens(p) for p in prons]
    act_tokens = phones_from_model(actual_phone_text)
    segs = align_tokens_dp(exp_tokens, act_tokens, _key_char)

    scores, heard = [], []
    for surf, pron, toks, seg in zip(eojeols, prons, exp_tokens, segs):
        errs = diff_tokens(toks, seg, _key_char)
        scores.append(
            score_from_errors(surf, item_type, pron, errs,
                              became_real_word=False, n_units=len(toks))
        )
        heard.append(phones_to_hangul([t.char for t in seg]))
    return scores, heard, eojeols


def format_sentence_report(scores, heard):
    lines = []
    n_ok = sum(1 for s in scores if s.correct)
    for s, h in zip(scores, heard):
        mark = "✅" if s.correct else "❌"
        lines.append(f"{mark} {s.word}  →  들린대로: {h or '(못 들음)'}   {s.accuracy:.0%}")
    total = len(scores)
    acc = n_ok / total if total else 0
    lines.append(f"\n어절 정확도: {n_ok}/{total} ({acc:.0%})")
    return "\n".join(lines)


def run_wav(sentence, wav_path):
    from phoneme_asr import transcribe_phones

    actual = transcribe_phones(wav_path)
    scores, heard, _ = score_sentence(sentence, actual)
    print(f"문장: {sentence}")
    print(f"들린 전체: 「{phones_to_hangul(actual)}」\n")
    print(format_sentence_report(scores, heard))
    return scores


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence", required=True)
    ap.add_argument("--wav")
    ap.add_argument("--dry")
    args = ap.parse_args()
    if args.dry:
        scores, heard, _ = score_sentence(args.sentence, args.dry)
        print(f"들린 전체: 「{phones_to_hangul(args.dry)}」\n")
        print(format_sentence_report(scores, heard))
    elif args.wav:
        run_wav(args.sentence, args.wav)
