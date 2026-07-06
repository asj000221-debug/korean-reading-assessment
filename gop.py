"""STEP 5 — GOP(Goodness of Pronunciation) 발음 신뢰도.

자유 인식(argmax)이 비슷한 음소로 틀려도(시→지), 모델이 '기대 음소'에
부여한 확률을 직접 보면 그 소리를 실제로 냈는지 알 수 있다.

방법: 기대 음소열을 음향에 CTC forced-align → 각 기대 음소의 프레임별
log-posterior 평균을 GOP로 사용. GOP가 높으면(0에 가까우면) 제대로 발음한 것.

용도(브리프 STEP 5): 하드룰 오류 중 GOP가 높은 것은 'ASR 미세오인식'으로 보고
'불확실(검수 필요)'로 강등 → false positive(제대로 읽었는데 오답) 감소.
"""

import numpy as np
import torch
from torchaudio.functional import forced_align

from asr import TARGET_SR, load_audio, preprocess
from phoneme_asr import _load


def _emission(speech):
    proc, model = _load()
    inputs = proc(speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # [1,T,V]
    return torch.log_softmax(logits, dim=-1), proc, model


def _blank_id(proc, model):
    bid = getattr(model.config, "pad_token_id", None)
    if bid is None:
        bid = proc.tokenizer.pad_token_id
    return 0 if bid is None else bid


def phone_gop(wav_path, expected_phones):
    """기대 음소열(list[str])을 정렬해 음소별 GOP(log prob) 반환.

    반환: list[(phone, gop)] — gop는 log-posterior 평균(높을수록 확신).
    정렬 실패 음소는 gop=None.
    """
    speech = preprocess(load_audio(wav_path))
    emission, proc, model = _emission(speech)  # [1,T,V]
    blank = _blank_id(proc, model)

    ids = proc.tokenizer.convert_tokens_to_ids(expected_phones)
    # OOV(없는 음소 기호)는 정렬에서 빠지지 않도록 unk로 대체
    unk = proc.tokenizer.unk_token_id or 0
    ids = [i if isinstance(i, int) and i >= 0 else unk for i in ids]
    targets = torch.tensor([ids], dtype=torch.int32)

    try:
        aligned, scores = forced_align(emission, targets, blank=blank)
    except Exception as e:
        return [(p, None) for p in expected_phones], f"align_fail:{e}"

    aligned = aligned[0].tolist()      # [T] 각 프레임에 배정된 토큰 id
    scores = scores[0].tolist()        # [T] 그 프레임 log prob
    # 토큰(비-blank) 등장 순서대로 GOP 집계
    per_tok, cur_id, buf = [], None, []
    # forced_align은 같은 음소가 여러 프레임 이어질 수 있음 → 연속 동일 토큰을 한 음소로
    # 단, 기대열에 같은 음소가 연달아 나오면 경계 모호 → 단순히 등장 순서로 채움
    seq_scores = [[] for _ in expected_phones]
    k = -1
    prev = blank
    for tid, sc in zip(aligned, scores):
        if tid == blank:
            prev = blank
            continue
        if tid != prev:           # 새 음소 시작
            k += 1
        if 0 <= k < len(expected_phones):
            seq_scores[k].append(sc)
        prev = tid

    out = []
    for p, ss in zip(expected_phones, seq_scores):
        out.append((p, float(np.mean(ss)) if ss else None))
    return out, "ok"


def gop_by_word(wav_path, expected_word_phone_tokens):
    """단어별 기대 음소 토큰 → 단어별 음소 GOP 리스트.

    expected_word_phone_tokens: list[list[Jamo]] (phoneme_map.to_phone_tokens 결과)
    반환: 단어별 list[(phone, gop)].
    """
    flat, bounds = [], []
    for toks in expected_word_phone_tokens:
        bounds.append((len(flat), len(flat) + len(toks)))
        flat.extend(t.char for t in toks)
    gops, status = phone_gop(wav_path, flat)
    return [gops[a:b] for (a, b) in bounds], status


# ── GOP 주채점 (argmax 대신 강제정렬 신뢰도로 직접 판정) ──
# 주의: 임계값은 '잠정값'이다. 정답 음소도 신뢰도가 넓게 퍼져(0~-3+) 단일 컷이
# 어렵다. 강한 증거(매우 낮은 GOP)만 오류로 보도록 보수적으로 둔다.
# 실제로는 임상가 채점본과 대조해 보정해야 함(브리프 §8).
GOP_HIGH = -3.0   # 이상 = 또렷이 발음(정답)
GOP_LOW = -6.0    # 미만 = 그 소리 없음(오류). 사이 = 불확실


def phone_state(gop):
    if gop is None:
        return "error"
    if gop >= GOP_HIGH:
        return "correct"
    if gop >= GOP_LOW:
        return "uncertain"
    return "error"


def score_word_gop(item_type, phone_tokens, gops):
    """기대 음소 토큰 + 음소별 GOP → (정확도, verdict, 음소별 상세).

    verdict: 'correct' | 'uncertain' | 'review'(오류 가능, 검수).
    transparent 종성(받침)은 관대(가중치 0).
    """
    n = max(len(phone_tokens), 1)
    werr, details = 0.0, []
    for tok, (p, g) in zip(phone_tokens, gops):
        st = phone_state(g)
        if item_type == "transparent" and tok.role == "coda":
            w = 0.0  # 받침 단순 약화 관대
        else:
            w = {"correct": 0.0, "uncertain": 0.5, "error": 1.0}[st]
        werr += w
        details.append((p, g, st, w))
    acc = max(0.0, 1.0 - werr / n)
    has_error = any(d[2] == "error" and d[3] > 0 for d in details)
    if werr == 0:
        verdict = "correct"
    elif has_error:
        verdict = "review"
    else:
        verdict = "uncertain"
    return acc, verdict, details


def score_list_gop(items, wav_path):
    """단어/어절 리스트 + wav → 단어별 (정확도, verdict, 상세, 저신뢰음소).

    items: list[{word,type}] (어절은 type 생략 시 transparent).
    """
    from g2p_expected import expected_prons
    from phoneme_map import to_phone_tokens

    etoks = []
    for it in items:
        t = it.get("type", "transparent")
        etoks.append(to_phone_tokens(expected_prons(it["word"], t)[0]))
    per_word_gop, status = gop_by_word(wav_path, etoks)

    out = []
    for it, toks, gl in zip(items, etoks, per_word_gop):
        t = it.get("type", "transparent")
        acc, verdict, details = score_word_gop(t, toks, gl)
        # 저신뢰(불확실/오류) 음소를 한글 초성/모음 기호로 표기
        weak = [(p, g, st) for (p, g, st, w) in details if st != "correct" and w > 0]
        out.append({"word": it["word"], "type": t, "accuracy": acc,
                    "verdict": verdict, "details": details, "weak": weak})
    return out, status


if __name__ == "__main__":
    import sys

    from phoneme_map import to_phone_tokens
    from g2p_expected import expected_prons

    wav = sys.argv[1] if len(sys.argv) > 1 else "recordings/list_001.wav"
    words = [("누마", "nonword"), ("디포", "nonword"), ("바누", "nonword"),
             ("꽃을", "phonrule"), ("옷이", "phonrule"), ("낮에", "phonrule"),
             ("바다", "transparent"), ("구두", "transparent"), ("머리", "transparent")]
    etoks = [to_phone_tokens(expected_prons(w, t)[0]) for w, t in words]
    per_word, status = gop_by_word(wav, etoks)
    print("status:", status)
    for (w, _), gl in zip(words, per_word):
        s = "  ".join(f"{p}:{g:.2f}" if g is not None else f"{p}:--" for p, g in gl)
        avg = np.mean([g for _, g in gl if g is not None]) if any(g is not None for _, g in gl) else None
        print(f"{w:5} avg={avg:.2f}  {s}" if avg is not None else f"{w:5} avg=--  {s}")
