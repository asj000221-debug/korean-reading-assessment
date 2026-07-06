"""STEP 4+ — 종단 학습자 프로파일 (회기 간 추적, 학습 0).

원 사례연구는 사실상 한 아동의 **152회기 종단 기록**(익명화)이다 — 회기마다
긍정/한계를 적고, 다음 개선방향을 정한다. 그러나 기존 시스템은 1회성 채점만 한다.
'난독증 진단/중재'의 본질은 **시간에 따른 변화**이므로 학습자 프로파일이 필수다.

이 모듈은 회기 결과를 누적 저장하고(JSON), 스킬별 숙달도 추이·오류유형 빈도 추이·
유창성(어절/분) 추이를 산출하며, 임상가가 손으로 쓰던 '학습결과(긍정/한계)'와
'개선방향'을 자동 생성한다. 규준(norm) 대조는 자리만 잡아둔다(규준표 필요).
"""

import json
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILE_DIR = os.path.join(_HERE, "profiles")


def _path(learner_id):
    return os.path.join(_PROFILE_DIR, f"{learner_id}.json")


def load(learner_id):
    """학습자 프로파일 로드(없으면 빈 프로파일)."""
    try:
        with open(_path(learner_id), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"learner_id": learner_id, "sessions": []}


def save(profile):
    os.makedirs(_PROFILE_DIR, exist_ok=True)
    with open(_path(profile["learner_id"]), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def append_session(learner_id, session_no, skill_scores, pattern_counts,
                   fluency=None, date=None):
    """회기 결과 1건을 프로파일에 추가하고 저장.

    skill_scores  : {skill_id: accuracy 0~1}   (이 회기에 측정된 스킬별 정확도)
    pattern_counts: {pattern: count}            (임상 오류유형 빈도)
    fluency       : {'eojeol_per_min': float}   (선택, 단락 유창성)
    date          : 'YYYY-MM-DD' (없으면 오늘)
    """
    profile = load(learner_id)
    profile["sessions"].append({
        "session_no": session_no,
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "skill_scores": skill_scores,
        "pattern_counts": pattern_counts,
        "fluency": fluency or {},
    })
    save(profile)
    return profile


def latest_evidence(profile):
    """스킬별 '가장 최근 측정 정확도'를 모아 placement용 evidence dict로."""
    ev = {}
    for s in profile["sessions"]:
        for sid, acc in s.get("skill_scores", {}).items():
            ev[sid] = acc  # 뒤 회기가 덮어씀 → 최신값
    return ev


def skill_trend(profile, skill_id):
    """특정 스킬의 회기별 (session_no, accuracy) 추이."""
    return [(s["session_no"], s["skill_scores"][skill_id])
            for s in profile["sessions"] if skill_id in s.get("skill_scores", {})]


def fluency_trend(profile):
    return [(s["session_no"], s["fluency"].get("eojeol_per_min"))
            for s in profile["sessions"] if s.get("fluency", {}).get("eojeol_per_min")]


def progress_note(profile, skill_labels=None):
    """임상가용 자동 '학습결과(긍정/한계)' + '추이' 생성.

    skill_labels: {skill_id: label} (없으면 skill_map에서 조회).
    """
    if skill_labels is None:
        try:
            from skill_map import SKILLS
            skill_labels = {k: v.label for k, v in SKILLS.items()}
        except Exception:
            skill_labels = {}

    sessions = profile["sessions"]
    if not sessions:
        return "(기록 없음)"
    last = sessions[-1]
    pos, lim = [], []
    for sid, acc in sorted(last.get("skill_scores", {}).items(), key=lambda kv: -kv[1]):
        label = skill_labels.get(sid, sid)
        # 추이: 이전 측정 대비 변화
        trend = skill_trend(profile, sid)
        delta = ""
        if len(trend) >= 2:
            d = trend[-1][1] - trend[-2][1]
            if abs(d) >= 0.05:
                delta = f" ({'▲' if d > 0 else '▼'}{abs(d):.0%})"
        line = f"{label} {acc:.0%}{delta}"
        (pos if acc >= 0.9 else lim).append(line)

    out = [f"■ {profile['learner_id']} — {last['session_no']}회기 ({last['date']})"]
    out.append("(긍정) " + ("; ".join(pos) if pos else "—"))
    out.append("(한계) " + ("; ".join(lim) if lim else "—"))
    ft = fluency_trend(profile)
    if ft:
        first, lastf = ft[0], ft[-1]
        out.append(f"(유창성) {first[1]:.0f}→{lastf[1]:.0f} 어절/분 "
                   f"[{first[0]}→{lastf[0]}회기]")
    # 최다 오류유형
    if last.get("pattern_counts"):
        top = sorted(last["pattern_counts"].items(), key=lambda kv: -kv[1])[:3]
        out.append("(주오류) " + ", ".join(f"{p}×{c}" for p, c in top))
    return "\n".join(out)


if __name__ == "__main__":
    # 시연: 가상의 종단 기록(임상 사례 사례 구조 모사) — profiles/_demo.json 생성
    lid = "_demo"
    if os.path.exists(_path(lid)):
        os.remove(_path(lid))
    append_session(lid, 9, {"decode_vowel_simple": 0.6, "pa_syllable": 0.8},
                   {"vowel_substitution": 3, "syllable_transposition": 2},
                   {"eojeol_per_min": 12}, date="2016-09-01")
    append_session(lid, 22, {"decode_vowel_simple": 0.85, "decode_vowel_glide": 0.5,
                             "pa_syllable": 0.9},
                   {"glide_simplification": 4}, {"eojeol_per_min": 18},
                   date="2016-12-01")
    append_session(lid, 55, {"decode_vowel_simple": 0.95, "decode_vowel_glide": 0.9,
                             "decode_onset": 0.9, "decode_coda": 0.85},
                   {"coda_deletion": 2, "sibilant_confusion": 3},
                   {"eojeol_per_min": 30}, date="2018-03-01")
    prof = load(lid)
    print(progress_note(prof))
    print("\n단모음 추이:", skill_trend(prof, "decode_vowel_simple"))
    print("유창성 추이:", fluency_trend(prof))
    print("\n최신 evidence(placement용):", latest_evidence(prof))
