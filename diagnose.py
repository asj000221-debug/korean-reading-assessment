"""STEP 4+ — 진단·처방 오케스트레이터 (채점 → 임상 리포트, 학습 0).

기존 파이프라인(asr→g2p→align→scoring)은 '정확도 %'에서 멈춘다. 이 모듈은 그 위에
임상 계층을 얹어, 한 회기 채점 결과를 다음으로 변환한다:

  채점 결과 ──▶ 임상 오류 프로파일 (error_taxonomy)
            ──▶ 약점 하위기술 + 스킬별 정확도(evidence)
            ──▶ 발달 배치 + 다음 목표 (skill_map.placement)
            ──▶ 개선방향(중재) 처방 (skill_map.prescribe)
            ──▶ 종단 프로파일 누적 (profile)

즉 '채점기'를 '진단·처방 엔진'으로 만드는 최종 결합부. 전부 규칙(학습 0).
"""

from collections import defaultdict

from scoring import score_word
from error_taxonomy import classify_word, classify_rule_failure, summarize_profile
from skill_map import placement, prescribe, SKILLS, format_ladder


def diagnose_session(results, real_words=None):
    """한 회기 채점 결과 → 임상 진단 dict.

    results: list[dict] — 각 항목:
        word        : 원본 철자
        expected    : 기대 발음(G2P 결과)
        actual      : 실제 발화(ASR/dry-run)
        item_type   : 'nonword'|'phonrule'|'transparent'
        skill       : (선택) skill_map 노드 id. 없으면 스킬별 집계에서 제외
        rule        : (선택) phonrule 규칙명

    반환 dict: word_results, clinical_errors, error_summary, evidence,
               placement, prescription
    """
    real_words = set(real_words or [])
    word_results, clinical = [], []
    skill_acc = defaultdict(list)  # {skill: [accuracy,...]}

    for r in results:
        ws = score_word(r["word"], r["item_type"], r["expected"], r["actual"],
                        real_words=real_words)
        ces = classify_word(r["expected"], r["actual"])
        # 음운변동 문항: 규칙 미적용 신호 추가
        if r["item_type"] == "phonrule" and r.get("rule"):
            applied = (r["actual"] == r["expected"])
            rf = classify_rule_failure(r["rule"], applied)
            if rf:
                ces.append(rf)
        clinical.extend(ces)
        if r.get("skill"):
            skill_acc[r["skill"]].append(ws.accuracy)
        word_results.append({
            "word": r["word"], "expected": r["expected"], "actual": r["actual"],
            "correct": ws.correct, "accuracy": ws.accuracy,
            "patterns": [c.pattern for c in ces],
        })

    summary = summarize_profile(clinical)
    evidence = {sid: sum(v) / len(v) for sid, v in skill_acc.items()}
    place = placement(evidence)
    rx = prescribe(summary["top"], evidence)

    return {
        "word_results": word_results,
        "clinical_errors": clinical,
        "error_summary": summary,
        "evidence": evidence,
        "placement": place,
        "prescription": rx,
    }


def format_report(dx):
    """진단 dict → 임상 리포트 텍스트."""
    L = []
    L.append("═" * 60)
    L.append("  난독증 읽기평가 — 임상 진단 리포트")
    L.append("═" * 60)

    # 1) 단어별 결과
    L.append("\n[1] 문항별 결과")
    for w in dx["word_results"]:
        mark = "O" if w["correct"] else "X"
        pat = ("  ⟶ " + ", ".join(w["patterns"])) if w["patterns"] else ""
        L.append(f"  [{mark}] {w['word']:8} {w['expected']}→{w['actual']:8} "
                 f"acc={w['accuracy']:.0%}{pat}")

    # 2) 오류 프로파일
    L.append("\n[2] 임상 오류 프로파일")
    pats = dx["error_summary"]["patterns"]
    if pats:
        for p, c in sorted(pats.items(), key=lambda kv: -kv[1]):
            L.append(f"  · {p}: {c}회")
    else:
        L.append("  · 유의 오류 없음")

    # 3) 약점 하위기술
    L.append("\n[3] 약점 하위기술(가중 심각도)")
    for sid, sc in dx["error_summary"]["top"][:5]:
        label = SKILLS[sid].label if sid in SKILLS else sid
        L.append(f"  · {label}: {sc:.1f}")

    # 4) 발달 배치
    pl = dx["placement"]
    L.append("\n[4] 발달 배치")
    L.append(f"  현재 단계 ▶ {pl.current_label}")
    L.append(f"  다음 목표 → " + ", ".join(
        SKILLS[t].label for t in pl.next_targets) if pl.next_targets else "  다음 목표 → —")

    # 5) 처방(개선방향)
    L.append("\n[5] 개선방향(처방)")
    for p in dx["prescription"][:4]:
        L.append(f"  · [{p['domain']}] {p['label']}  ({p['ppt']})")
        L.append(f"      → {p['method']}")
        if p["note"]:
            L.append(f"      ⚠ {p['note']}")
    L.append("═" * 60)
    return "\n".join(L)


if __name__ == "__main__":
    # 시연: 사례연구 초·중기 오류를 반영한 가상 회기 채점 결과
    session = [
        # 단모음 단계: 일부 정답, 일부 모음 대치
        {"word": "고기", "expected": "고기", "actual": "고기", "item_type": "transparent",
         "skill": "decode_vowel_simple"},
        {"word": "거미", "expected": "거미", "actual": "고미", "item_type": "transparent",
         "skill": "decode_vowel_simple"},
        # 미끄러지는 모음: 활음 단순화
        {"word": "야구", "expected": "야구", "actual": "아구", "item_type": "transparent",
         "skill": "decode_vowel_glide"},
        {"word": "교회", "expected": "교회", "actual": "고회", "item_type": "transparent",
         "skill": "decode_vowel_glide"},
        # 치찰음 혼동
        {"word": "자두", "expected": "자두", "actual": "사두", "item_type": "transparent",
         "skill": "percept_sibilant"},
        # 초성 생략(음소수준 약점)
        {"word": "가죽", "expected": "가죽", "actual": "아죽", "item_type": "transparent",
         "skill": "pa_phoneme"},
        # 음운변동 미적용
        {"word": "국물", "expected": "궁물", "actual": "국물", "item_type": "phonrule",
         "skill": "rule_nasal", "rule": "비음화"},
    ]
    dx = diagnose_session(session)
    print(format_report(dx))
