# -*- coding: utf-8 -*-
"""③ 음운인식(합성) — 닫힌집합(closed-set) 검증.

매뉴얼 원칙은 "목표 후보 안에서 인식 → 말한 게 목표와 같은지 판정"이다. 낱말/단락의
열린 전사와 패러다임이 다르다. 후보가 16개로 닫혀 있으니 자유 디코딩(argmax) 대신
후보 제약을 건다: 각 후보 음소열을 오디오에 CTC forced-align → Viterbi 경로 log-우도
→ 최우도 후보 선택 → 목표와 같으면 정답.

짧은 무의미 음절(따·규…2음소)은 자유 디코딩이면 오인식이 잦은데, 후보를 16개로
가두면 "그 중 뭐냐"만 풀면 되니 훨씬 강건하다. 낱말/단락 음향모델을 그대로 쓰되
디코딩만 제약, 파인튜닝 없이 baseline.

verify(wav)는 실제 오디오, verify_dry(phones)는 음소열 문자열로 모델 없이 로직만 점검.
"""

from phoneme_map import phone_str, phones_from_model, to_phone_tokens
from task_db import pa_candidates, pa_items


def _cand_phone_seqs():
    """후보 16개 → (단어, 음소열 list[str]) 리스트."""
    return [(c, [t.char for t in to_phone_tokens(c)]) for c in pa_candidates()]


# ── 오디오 경로: forced-align Viterbi 우도로 후보 선택 ──
def _emission_once(wav_path):
    from asr import load_audio, preprocess
    from gop import _emission
    speech = preprocess(load_audio(wav_path))
    return _emission(speech)  # (log_softmax emission [1,T,V], proc, model)


def _path_logprob(emission, proc, model, phones):
    """후보 음소열의 forced-align Viterbi 경로 평균 log-우도(프레임당). 실패 시 None."""
    import torch
    from torchaudio.functional import forced_align
    from gop import _blank_id

    blank = _blank_id(proc, model)
    unk = proc.tokenizer.unk_token_id or 0
    ids = proc.tokenizer.convert_tokens_to_ids(phones)
    ids = [i if isinstance(i, int) and i >= 0 else unk for i in ids]
    if not ids:
        return None
    targets = torch.tensor([ids], dtype=torch.int32)
    try:
        _aligned, scores = forced_align(emission, targets, blank=blank)
    except Exception:
        return None
    T = emission.shape[1] or 1
    return float(scores[0].sum()) / T  # 프레임당 평균(후보 간 공정 비교)


# 목표 음소 최저 GOP가 이 값 미만이면 '그 음소를 안 냈다' → 오발음(오답).
# 잠정값이라 임상 채점본과 대조해 보정해야 한다.
GOP_OK = -3.0


def verify(wav_path, target=None, gop_ok=GOP_OK):
    """오디오 → 닫힌집합 후보 우도 순위 + 판정.

    닫힌집합 식별(pred = 최우도 후보)과 오발음 판정(correct)을 분리한다:
      pred    : 16후보 중 아이가 시도한 것으로 가장 그럴듯한 후보(항상 하나 선택됨).
      correct : pred==target 이고, 목표 음소를 실제로 냈는지(최저 GOP≥임계) → 매뉴얼
                "잘못 발음하면 오답". GOP가 낮으면 최근접이 목표여도 오답(오발음).

    반환 dict: pred, correct, score, margin, min_gop, reject, ranking(top5).
    """
    emission, proc, model = _emission_once(wav_path)
    scored = []
    for word, phones in _cand_phone_seqs():
        lp = _path_logprob(emission, proc, model, phones)
        if lp is not None:
            scored.append((word, lp))
    scored.sort(key=lambda x: x[1], reverse=True)
    res = _finish(scored, target)

    # 오발음 게이트: 목표 음소열의 최저 GOP로 '정말 그 소리를 냈는지' 확인
    min_gop, reject = None, None
    if target is not None:
        from gop import phone_gop
        tphones = [t.char for t in to_phone_tokens(target)]
        gl, _st = phone_gop(wav_path, tphones)
        gs = [g for _p, g in gl if g is not None]
        min_gop = round(min(gs), 3) if gs else None
        if res["pred"] != target:
            reject = "wrong_candidate"          # 다른 후보로 합성(합성 실패)
        elif min_gop is not None and min_gop < gop_ok:
            reject = "mispronounced"            # 목표는 맞췄으나 음소 오발음
        res["correct"] = reject is None
    res["min_gop"], res["reject"] = min_gop, reject
    return res


# ── dry 경로: slplab 음소열 → 후보 중 음소 최근접(모델 없이 로직 점검) ──
def _lev(a, b):
    """음소 토큰열 편집거리."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def verify_dry(actual_phone_text, target=None):
    """slplab 음소열 문자열 → 후보 중 음소 편집거리 최소 후보 선택(모델 없이 점검).

    pred    : 최근접 후보(닫힌집합 식별). score=-편집거리, margin=1위-2위 차.
    correct : 목표 음소열과 편집거리 0(음소 정확 산출) → 매뉴얼 "잘못 발음하면 오답".
    reject  : 'wrong_candidate'(다른 후보) | 'mispronounced'(목표 맞췄으나 음소오류).
    """
    act = [t.char for t in phones_from_model(actual_phone_text)]
    scored = [(w, -float(_lev(act, ph))) for w, ph in _cand_phone_seqs()]
    scored.sort(key=lambda x: x[1], reverse=True)
    res = _finish(scored, target)
    if target is not None:
        tphones = [t.char for t in to_phone_tokens(target)]
        dist = _lev(act, tphones)
        if res["pred"] != target:
            reject = "wrong_candidate"
        elif dist > 0:
            reject = "mispronounced"
        else:
            reject = None
        res["correct"] = reject is None
        res["reject"] = reject
    return res


def _finish(scored, target):
    if not scored:
        return {"pred": None, "correct": None, "score": None, "margin": None,
                "reject": None, "ranking": []}
    pred, best = scored[0]
    margin = best - scored[1][1] if len(scored) > 1 else None
    return {
        "pred": pred,
        "correct": (pred == target) if target is not None else None,
        "score": round(best, 4),
        "margin": round(margin, 4) if margin is not None else None,
        "reject": None,
        "ranking": [(w, round(s, 4)) for w, s in scored[:5]],
    }


# ── 배치 채점: 여러 문항 → 2-3음소 / 4-5음소 정답률 ──
def score_all(results):
    """results: list[{"item": pa_item, "verify": verify(...) dict}] → 구간별 정답률."""
    agg = {"2-3": [0, 0], "4-5": [0, 0]}  # [correct, total]
    for r in results:
        b = r["item"]["bin"]
        agg[b][1] += 1
        if r["verify"].get("correct"):
            agg[b][0] += 1
    out = {}
    for b, (c, t) in agg.items():
        out[b] = {"correct": c, "total": t, "rate": (c / t) if t else 0.0}
    tot_c = sum(v["correct"] for v in out.values())
    tot_t = sum(v["total"] for v in out.values())
    out["overall"] = {"correct": tot_c, "total": tot_t, "rate": (tot_c / tot_t) if tot_t else 0.0}
    return out


if __name__ == "__main__":
    # 모델 없이 dry 로직 점검: 각 타깃을 '정확히 발음'한 음소열을 만들어 넣으면
    # 후보 제약이 100% 그 타깃을 골라야 한다. 그리고 오발음(대치) 케이스 하나.
    items = pa_items()
    print("=== dry: 각 타깃 정발음 음소열 → 닫힌집합 선택 ===")
    results = []
    for it in items:
        phones = phone_str(to_phone_tokens(it["target"]))
        v = verify_dry(phones, target=it["target"])
        mark = "✅" if v["correct"] else "❌"
        results.append({"item": it, "verify": v})
        print(f"{mark} {it['target']:4}({it['bin']}) 음소=[{phones:14}] → 선택={v['pred']:4} margin={v['margin']}")
    print("\n구간별 정답률:", score_all(results))

    print("\n=== dry: 오발음/오합성 케이스 ===")
    for got, tgt, desc in [("G I D EO", "기떠", "기더(DD→D 대치, 목표 맞춤·음소오류)"),
                            ("Kh A N I", "기떠", "카니로 합성(다른 후보)"),
                            ("G I DD EO", "기떠", "정발음")]:
        v = verify_dry(got, target=tgt)
        print(f"목표 {tgt} / 발화[{got:9}] {desc:28} → pred={v['pred']:4} "
              f"correct={v['correct']} reject={v['reject']}")
