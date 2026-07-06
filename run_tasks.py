# -*- coding: utf-8 -*-
"""3개 ASR 과제 통합 러너 — task_db(정답 DB)에 물려 돌린다.

과제별로 모델/인식단위/채점을 한 진입점으로 묶었다.
  ① 낱말읽기: slplab 음소-CTC 열린전사 → 단어별 자모 diff → 유형별 엄격도(무의미 최엄격)
  ② 단락읽기: 같은 음향모델 재사용 → P1 어절 정렬 → 어절 정확도
  ③ 음운인식: 닫힌집합(16후보) 검증(pa_synth) — 별도 모델 아니고 디코딩만 제약

세 과제가 음향모델을 공유한다(파인튜닝은 낱말읽기 하나에 집중). --wav은 실제
오디오, --dry는 음소열을 직접 넣어 모델 없이 채점 로직만 점검.
"""

import re

from align import _key_char, align_tokens_dp, diff_tokens
from phoneme_map import phones_from_model, phones_to_hangul, phone_str, to_phone_tokens
from scoring import score_from_errors
import task_db

_PUNCT = re.compile(r"[.,!?…·()]")
_QUOTE = re.compile(r"[\"'“”‘’]")


# ══════════════ ① 낱말읽기 ══════════════
def score_words(actual_phone_text, section="all"):
    """연결발화 음소열 → 단어별 채점. 반환: (items, scores, heard)."""
    items = task_db.word_items(section)
    exp_tokens = [to_phone_tokens(it["pron"]) for it in items]
    act = phones_from_model(actual_phone_text)
    segs = align_tokens_dp(exp_tokens, act, _key_char)

    scores, heard = [], []
    for it, toks, seg in zip(items, exp_tokens, segs):
        errs = diff_tokens(toks, seg, _key_char)
        h = phones_to_hangul([t.char for t in seg])
        # transparent만 '다른 실단어로 바뀜' 판정(종성 관대 해제)
        became = (
            it["type"] == "transparent"
            and h != it["word"]
            and h in task_db.REAL_WORDS
        )
        scores.append(
            score_from_errors(it["word"], it["type"], it["pron"], errs,
                              became_real_word=became, n_units=len(toks))
        )
        heard.append(h)
    return items, scores, heard


def summarize_words(items, scores):
    """의미/무의미 구간별 정답률 요약."""
    agg = {"의미": [0, 0], "무의미": [0, 0]}
    for it, s in zip(items, scores):
        agg[it["section"]][1] += 1
        if s.correct:
            agg[it["section"]][0] += 1
    out = {}
    for sec, (c, t) in agg.items():
        if t:
            out[sec] = {"correct": c, "total": t, "rate": c / t}
    tc = sum(v["correct"] for v in out.values())
    tt = sum(v["total"] for v in out.values())
    out["전체"] = {"correct": tc, "total": tt, "rate": (tc / tt) if tt else 0.0}
    return out


def report_words(items, scores, heard):
    lines = []
    for it, s, h in zip(items, scores, heard):
        mark = "✅" if s.correct else "❌"
        lines.append(f"{mark} {it['no']:>2} [{it['section']:3}/{s.item_type:11}] "
                     f"{it['word']:6} 기대={it['pron']:6} 들림={h or '(못들음)':6} acc={s.accuracy:.0%}")
    s = summarize_words(items, scores)
    lines.append("\n── 정답률 ──")
    for k, v in s.items():
        lines.append(f"  {k}: {v['correct']}/{v['total']} ({v['rate']:.0%})")
    return "\n".join(lines)


# ══════════════ ② 단락읽기 ══════════════
def _para_eojeols():
    """P1 → (표기 어절, 기대발음 어절) 리스트. 문장 단위 g2p(어절경계·연음 보존)."""
    from g2p_expected import g2p
    txt = _QUOTE.sub("", task_db.paragraph()["text"])
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", txt) if s.strip()]
    surf, prons = [], []
    for s in sents:
        surf += [w for w in _PUNCT.sub("", s).split() if w]
        prons += [w for w in _PUNCT.sub("", g2p(s)).split() if w]
    n = min(len(surf), len(prons))
    return surf[:n], prons[:n]


def score_paragraph(actual_phone_text):
    """연결발화 음소열 → 어절별 채점. 반환: (surf, scores, heard)."""
    surf, prons = _para_eojeols()
    exp_tokens = [to_phone_tokens(p) for p in prons]
    act = phones_from_model(actual_phone_text)
    segs = align_tokens_dp(exp_tokens, act, _key_char)
    scores, heard = [], []
    for w, toks, seg in zip(surf, exp_tokens, segs):
        errs = diff_tokens(toks, seg, _key_char)
        scores.append(score_from_errors(w, "transparent", w, errs,
                                        became_real_word=False, n_units=len(toks)))
        heard.append(phones_to_hangul([t.char for t in seg]))
    return surf, scores, heard


def report_paragraph(surf, scores):
    n_ok = sum(1 for s in scores if s.correct)
    tot = len(scores)
    bad = [f"{s.word}" for s in scores if not s.correct]
    lines = [f"어절 정확도: {n_ok}/{tot} ({(n_ok/tot if tot else 0):.0%})"]
    if bad:
        lines.append("오독 어절: " + " ".join(bad[:40]) + (" …" if len(bad) > 40 else ""))
    return "\n".join(lines)


# ══════════════ ③ 음운인식(합성) ══════════════
def run_pa_dry(pairs):
    """pairs: list[(target, actual_phone_text)] → 결과 리스트 + 구간별 정답률."""
    import pa_synth
    items = {it["target"]: it for it in task_db.pa_items()}
    results = []
    for target, phones in pairs:
        v = pa_synth.verify_dry(phones, target=target)
        results.append({"item": items[target], "verify": v})
    return results, pa_synth.score_all(results)


# ══════════════ 셀프테스트(모델 없이 로직 검증) ══════════════
def _selftest():
    print("═══ 셀프테스트: 완벽 발화 dry → 100% 나와야 정상 ═══\n")

    # ① 낱말: 각 단어를 기대발음 그대로 → 연결 음소열
    items = task_db.word_items("all")
    perfect = " ".join(phone_str(to_phone_tokens(it["pron"])) for it in items)
    its, scores, heard = score_words(perfect)
    s = summarize_words(its, scores)
    print("① 낱말읽기(완벽):", {k: f"{v['correct']}/{v['total']}" for k, v in s.items()})

    # 오독 주입: 무의미 '낚씨'를 '낙시'(경음화 실패)로 → 무의미 1개 오답 기대
    idx = next(i for i, it in enumerate(items) if it["word"] == "낚씨")
    toks = [phone_str(to_phone_tokens(it["pron"])) for it in items]
    toks[idx] = toks[idx].replace("SS", "S")  # SS→S
    its, scores, _ = score_words(" ".join(toks))
    nono = summarize_words(its, scores)["무의미"]
    print(f"   무의미 '낚씨'→'낙시'(경음화실패) 주입: 무의미 {nono['correct']}/{nono['total']} "
          f"(39/40이면 정상: 경음화 실패를 오답 처리)")

    # ② 단락: 완벽 발화 → 100%
    surf, prons = _para_eojeols()
    perfect_p = " ".join(phone_str(to_phone_tokens(p)) for p in prons)
    surf, scores, _ = score_paragraph(perfect_p)
    n_ok = sum(1 for s in scores if s.correct)
    print(f"② 단락읽기(완벽): {n_ok}/{len(scores)} 어절")

    # ③ 음운인식: 완벽/오합성/오발음
    pairs = [(it["target"], phone_str(to_phone_tokens(it["target"])))
             for it in task_db.pa_items()]
    _res, agg = run_pa_dry(pairs)
    print("③ 음운인식(완벽):", {k: f"{v['correct']}/{v['total']}" for k, v in agg.items()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="매뉴얼 3과제 통합 러너")
    sub = ap.add_subparsers(dest="task")

    pw = sub.add_parser("words", help="① 낱말읽기")
    pw.add_argument("--wav"); pw.add_argument("--dry")
    pw.add_argument("--section", default="all", choices=["all", "meaning", "nonsense"])

    pp = sub.add_parser("para", help="② 단락읽기")
    pp.add_argument("--wav"); pp.add_argument("--dry")

    pa = sub.add_parser("pa", help="③ 음운인식(합성)")
    pa.add_argument("--wav"); pa.add_argument("--target")

    sub.add_parser("info", help="DB 요약")
    sub.add_parser("selftest", help="모델 없이 채점 로직 검증")

    args = ap.parse_args()

    if args.task == "info":
        import runpy
        runpy.run_path("task_db.py", run_name="__main__")
    elif args.task == "selftest":
        _selftest()
    elif args.task == "words":
        if args.wav:
            from phoneme_asr import transcribe_phones
            actual = transcribe_phones(args.wav)
        else:
            actual = args.dry or ""
        its, scores, heard = score_words(actual, args.section)
        print(f"들린 전체: 「{phones_to_hangul(actual)}」\n")
        print(report_words(its, scores, heard))
    elif args.task == "para":
        if args.wav:
            from phoneme_asr import transcribe_phones
            actual = transcribe_phones(args.wav)
        else:
            actual = args.dry or ""
        surf, scores, heard = score_paragraph(actual)
        print(report_paragraph(surf, scores))
    elif args.task == "pa":
        if not args.target:
            ap.error("pa 과제는 --target 필요 (예: --target 기떠)")
        if args.wav:
            import pa_synth
            v = pa_synth.verify(args.wav, target=args.target)
        else:
            ap.error("현재 pa 단건은 --wav 필요(또는 selftest로 dry 검증)")
        print(f"목표={args.target} → pred={v['pred']} correct={v['correct']} "
              f"reject={v['reject']} min_gop={v.get('min_gop')}\n순위: {v['ranking']}")
    else:
        ap.print_help()
