"""발달 위계 스킬맵 + 배치/처방 엔진(규칙 기반).

임상 사례연구의 치료절차는 평면이 아니라 위계다. 음운인식·해독·쓰기 3영역이
병렬로 가되, 각 영역 안에서 쉬운 것부터 어려운 것 순으로 전제조건을 이루며 쌓인다.
아동이 막힌 지점을 찾아 다음 목표를 처방하려면 이 위계를 그래프로 박아둬야 한다.

- SKILLS: 치료절차를 노드(하위기술) DAG로 인코딩
- placement(): 오류/정확도 증거로 '막힌 첫 노드' 산출
- prescribe(): 약점 노드 → 다음 목표 + 사례 근거 중재법 처방

노드 id는 error_taxonomy가 가리키는 약점 스킬과 맞춰 뒀다.
"""

from collections import namedtuple

# 노드 = 하위기술. level은 발달 순서(작을수록 먼저, 배치/정렬 기준),
# prereqs는 숙달돼야 진입 가능한 선행 노드, method는 약할 때의 중재법(처방 텍스트).
Skill = namedtuple("Skill", ["id", "domain", "level", "label", "prereqs", "ppt", "method"])

_RAW = [
    # ── 음운인식 (PA) : 글자 없는 청각 과제 ──
    ("pa_syllable", "음운인식", 10, "음절수준 음운인식(수세기·분절·합성·생략·대치·도치)",
     [], "사례 관찰 회기2-22",
     "바둑돌/박수로 음절 수세기, 합성·생략·도치 구두 과제. 도치(가방→바강) 오류 시 음절 거꾸로말하기 집중."),
    ("percept_consonant", "음운인식", 12, "자음 청각변별",
     [], "사례 관찰 회기2-9",
     "최소대립쌍 듣고 변별. 소리값 기억 어려우면 입모양 사진 힌트(visual phonics 아닌 입모양만, 사례 관찰)."),
    ("percept_sibilant", "음운인식", 14, "치찰음 변별(ㅅ↔ㅈ↔ㅊ)",
     ["percept_consonant"], "사례 관찰 청각변별 최약점",
     "자두/사두 류 ㅅ↔ㅈ 최소대립쌍 집중 변별. 말소리 샘플 정확도 자체가 낮으면 조음 점검 병행."),
    ("pa_phoneme", "음운인식", 20, "음소수준 음운인식(cv 인지·분절·합성·생략·첨가·대치)",
     ["pa_syllable"], "사례 관찰 회기17+",
     "ㅂ+ㅓ→버 합성/분절. 초성 생략(가죽→아욱) 오류 시 '첫 소리 빼기' 음소 조작 집중."),

    # ── 해독 (decode) : 모음 라인 ──
    ("decode_vowel_simple", "해독", 11, "단모음 해독(ㅏㅓㅗㅜㅡㅣㅐㅔ)",
     ["percept_consonant"], "사례 관찰 회기2-9",
     "입모양 사진으로 소리값 연결, 자석판으로 글자 만들고 목표모음 확인(사례 관찰)."),
    ("decode_vowel_glide", "해독", 16, "미끄러지는 모음 해독(ㅑㅕㅛㅠㅒㅖ)",
     ["decode_vowel_simple"], "사례 관찰 회기10-16",
     "이→아 미끄러지기 시범(/이.아/→/야/). 활음 단순화(야→아) 오류 시 단모음 2개 빠르게 합성 연습."),
    ("decode_vowel_diphthong", "해독", 21, "이중모음 해독(ㅘㅝㅙㅞㅚㅟㅢ)",
     ["decode_vowel_glide"], "사례 관찰 회기17-22",
     "모음카드 Quick Drill. 웨↔워 혼동/활음 첨가(카→콰) 시 w계 최소대립쌍(사례 관찰) 오류극복 과제."),

    # ── 해독 (decode) : 자음/합성/받침 라인 ──
    ("decode_onset", "해독", 30, "첫소리 자음 해독(조음가족별)",
     ["pa_phoneme", "decode_vowel_simple"], "사례 관찰 회기23-44",
     "Slow 합성 시범(청각 먼저), 글자 보여주고 없애며 부드럽게 합성(사례 관찰). 조음가족(입술뻥뻥 등) 단위."),
    ("decode_blend", "해독", 32, "CV 합성(자음+모음 끊지 않고 잇기)",
     ["decode_onset"], "사례 관찰 회기23+",
     "ㅅ+ㅏ→사 멈추지 않고 합성. 그림 제시로 가능성 추론(의미단어)."),
    ("decode_coda", "해독", 45, "끝소리 자음 해독(받침)",
     ["decode_blend"], "사례 관찰 회기45-55",
     "CVC 합성(ㅌㅗㅇ→통). 받침 인식(-) 시 끝소리 변별 그림과제(사례 관찰) 선행."),
    ("rule_final7", "해독", 56, "끝소리규칙(7종성 중화)",
     ["decode_coda"], "사례 관찰 회기56-89",
     "약/벽/확/엮 → 받침 중화 규칙. 무의미 1음절 cvc로 규칙 일반화 확인."),
    ("decode_nonword", "해독", 60, "비단어 해독(순수 해독력)",
     ["rule_final7"], "사례 관찰 회기56-89",
     "비단어 매일 읽기. 추측읽기 없이 자소-음소 변환만으로 읽는지 = 난독 변별 1순위."),
    ("fluency_paragraph", "해독", 90, "단락 읽기 유창성(어절/분)",
     ["decode_nonword"], "사례 관찰 회기90+",
     "초 재며 단락 읽기, 어절수/60초 추적. 정확도 확보 후 속도(자동화) 목표."),

    # ── 해독 (decode) : 음운변동 10종 (회기108-152, 사례 관찰) ──
    ("rule_liaison", "해독", 108, "연음규칙",
     ["rule_final7"], "회기108-123",
     "꽃을→꼬츨. 받침이 뒤 모음으로 넘어가 읽힘. 형태소 경계 인식과 연계."),
    ("rule_aspirate", "해독", 124, "격음화(축약)",
     ["rule_liaison"], "회기124-133", "좋다→조타, 축하→추카. ㅎ+평음→격음 축약."),
    ("rule_hdrop", "해독", 134, "ㅎ 탈락",
     ["rule_aspirate"], "회기134", "많아→마나. 받침 ㅎ이 모음 앞에서 탈락."),
    ("rule_cluster", "해독", 135, "겹받침",
     ["rule_hdrop"], "회기135-140", "읽다→익따, 앉다→안따. 겹받침 대표음 선택 + 경음화."),
    ("rule_tense", "해독", 141, "경음화",
     ["rule_cluster"], "회기141-144", "국밥→국빱, 숙제→숙쩨. 받침 뒤 평음 된소리화."),
    ("rule_nasal", "해독", 145, "비음화",
     ["rule_tense"], "회기145-147", "국물→궁물, 밥물→밤물. 장애음+비음→비음."),
    ("rule_lateral", "해독", 146, "유음화(설측음화)",
     ["rule_tense"], "회기145-147", "신라→실라, 난로→날로. ㄴ+ㄹ→ㄹㄹ."),
    ("rule_saisiot", "해독", 148, "음소첨가/사이시옷",
     ["rule_nasal"], "회기148-150", "ㄴ첨가/사이시옷. 합성어 경계 소리 첨가."),
    ("rule_palatal", "해독", 151, "구개음화",
     ["rule_saisiot"], "회기151-152", "해돋이→해도지, 같이→가치. ㄷㅌ+이→ㅈㅊ."),

    # ── 쓰기 (write) ──
    ("write_phoneme", "쓰기", 25, "음소분절 받아쓰기",
     ["pa_phoneme"], "사례 관찰 회기23+",
     "소리를 음소로 나눠 받아쓰기. 2-3음절 반복 제시 필요 시 음소 인식 보강."),
    ("write_phonetic", "쓰기", 58, "소리나는대로 받아쓰기",
     ["write_phoneme", "rule_final7"], "사례 관찰 회기56-89",
     "들리는 표면형 그대로 표기. 표기인식 검사 선행."),
    ("write_ortho", "쓰기", 100, "표기인식/바르게 쓰기",
     ["write_phonetic"], "사례 관찰 회기92+",
     "숟까락 vs 숟가락 비교재인. 소리나는대로 쓴 문장 고치기."),
    ("write_rule", "쓰기", 110, "음운변동 철자(기본형 찾기)",
     ["write_ortho", "rule_liaison"], "사례 관찰 회기108+",
     "/무더요/→묻다(기본형). 존댓말↔반말 만들며 받침 역추론."),
]

SKILLS = {s[0]: Skill(*s) for s in _RAW}

MASTERY_THRESHOLD = 0.9  # 정확도 이 이상이면 숙달

PlacementResult = namedtuple(
    "PlacementResult", ["current", "current_label", "domain_levels", "next_targets", "mastered"]
)


def node(skill_id):
    return SKILLS.get(skill_id)


def _is_mastered(skill_id, evidence):
    """evidence[skill_id]가 임계 이상이면 숙달. 미검사(None/없음)는 미숙달로 보지 않고 보류."""
    v = evidence.get(skill_id)
    return v is not None and v >= MASTERY_THRESHOLD


def _prereqs_met(skill, evidence):
    return all(_is_mastered(p, evidence) for p in skill.prereqs)


def placement(evidence):
    """발달 배치: 증거(evidence)로 '현재 막힌 단계'와 다음 목표를 산출.

    evidence: {skill_id: mastery_score 0~1}  (검사 안 한 노드는 생략 가능)

    현재 단계 = 전제조건은 충족됐는데 본인은 미숙달인 가장 낮은 level 노드.
    """
    mastered = {sid for sid in SKILLS if _is_mastered(sid, evidence)}
    # 현재 막힌 단계 = 검사됐고 미숙달인 가장 낮은 level 노드(전제조건과 무관 —
    # '어디서 막혔나'를 기술). 전제조건 게이팅은 next_targets(가르칠 준비된 것)에만.
    tested_unmastered = [
        s for s in SKILLS.values() if s.id in evidence and s.id not in mastered
    ]
    tested_unmastered.sort(key=lambda s: s.level)
    current = tested_unmastered[0] if tested_unmastered else None

    # 다음 목표: 현재 + 전제조건 충족된 미숙달 노드들(level 순) 상위 3
    nexts = [s for s in SKILLS.values()
             if s.id not in mastered and _prereqs_met(s, evidence)]
    nexts.sort(key=lambda s: s.level)
    next_targets = [s.id for s in nexts[:3]]

    # 영역별 도달 level(숙달한 노드 중 최고 level)
    domain_levels = {}
    for s in SKILLS.values():
        if s.id in mastered:
            domain_levels[s.domain] = max(domain_levels.get(s.domain, 0), s.level)

    return PlacementResult(
        current.id if current else None,
        current.label if current else "—",
        domain_levels, next_targets, sorted(mastered),
    )


def prescribe(weak_skills, evidence=None):
    """약점 스킬(가중 점수 내림차순) → 처방 리스트.

    weak_skills: [(skill_id, score)]  (error_taxonomy.summarize_profile()['top'])
    evidence   : 선택. 있으면 전제조건 미충족 약점은 '선행기술 먼저'로 표시.

    반환: [{skill, label, domain, ppt, method, note}]  (발달 순서대로 정렬)
    """
    evidence = evidence or {}
    out = []
    seen = set()
    # 약점을 level 순으로 — 낮은(기초) 약점부터 다뤄야 위로 쌓인다
    weak_sorted = sorted(
        (sid for sid, _ in weak_skills if sid in SKILLS),
        key=lambda sid: SKILLS[sid].level,
    )
    for sid in weak_sorted:
        if sid in seen:
            continue
        seen.add(sid)
        s = SKILLS[sid]
        note = ""
        if not _prereqs_met(s, evidence) and s.prereqs:
            unmet = [SKILLS[p].label for p in s.prereqs if not _is_mastered(p, evidence)]
            note = "선행 미충족: " + ", ".join(unmet)
        out.append({"skill": s.id, "label": s.label, "domain": s.domain,
                    "ppt": s.ppt, "method": s.method, "note": note})
    return out


def format_ladder(evidence):
    """발달 사다리를 텍스트로(숙달 ●/현재 ▶/미검사 ·)."""
    pl = placement(evidence)
    lines = []
    for domain in ("음운인식", "해독", "쓰기"):
        lines.append(f"[{domain}]")
        for s in sorted((x for x in SKILLS.values() if x.domain == domain),
                        key=lambda x: x.level):
            if s.id in pl.mastered:
                mark = "●"
            elif s.id == pl.current:
                mark = "▶"
            elif s.id in evidence:
                mark = "△"  # 검사됨·미숙달
            else:
                mark = "·"  # 미검사
            sc = evidence.get(s.id)
            scs = f" ({sc:.0%})" if sc is not None else ""
            lines.append(f"  {mark} {s.label}{scs}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 시연: 사례 초기(단모음 단계에서 막힘)를 가정한 증거
    ev = {
        "pa_syllable": 0.85,        # 도치 약함
        "percept_consonant": 0.7,
        "percept_sibilant": 0.5,    # 최약점
        "decode_vowel_simple": 0.6,
        "decode_vowel_glide": 0.3,
    }
    pl = placement(ev)
    print("현재 발달 단계:", pl.current_label)
    print("다음 목표:", [SKILLS[t].label for t in pl.next_targets])
    print()
    print(format_ladder(ev))
    print()
    weak = [("percept_sibilant", 6.0), ("decode_vowel_glide", 3.0),
            ("pa_syllable", 3.0)]
    print("=== 처방(개선방향) ===")
    for p in prescribe(weak, ev):
        print(f"· [{p['domain']}] {p['label']}  ({p['ppt']})")
        print(f"    → {p['method']}")
        if p["note"]:
            print(f"    ⚠ {p['note']}")
