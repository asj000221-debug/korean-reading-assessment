"""STEP 4+ — 임상 오류유형 분류기 (난독증 진단 신호 추출, 학습 0).

현재 시스템은 오류를 대치/생략/첨가(substitution/omission/insertion)로만 본다.
그러나 임상가(SLP)는 *어떤 종류의* 오류인지를 본다 — 초성 생략인지, 음절 도치인지,
활음(미끄러지는 모음) 탈락인지, ㅅ↔ㅈ 청각변별 혼동인지. 오류의 *질*이 곧 진단 신호다.

이 모듈은 align.py의 raw Error(자모/음소 차이)에 **조음·음운 자질표**를 적용해
난독증 임상에서 실제 관찰되는 오류유형으로 분류한다. 전부 규칙(학습 0).

분류 카테고리는 익명화된 임상 사례연구의 실관찰 오류에 근거한다:
  onset_deletion          초성 생략        가죽→아욱, 까마귀→아아귀   (사례 관찰)
  coda_deletion           종성 생략        멀튼→머튼, 판→파            (사례 관찰)
  coda_substitution       종성 대치/중화   끝소리규칙 오류              (사례 관찰)
  syllable_transposition  음절/자모 도치   가방→바강                   (사례 관찰)
  glide_simplification    활음 단순화      야→아, 이.아 미끄러지기 실패 (사례 관찰)
  glide_insertion         활음 첨가        카→콰, 키→퀴                (사례 관찰)
  w_diphthong_confusion   w계 모음 혼동    웨↔워, 외↔왜↔웨            (사례 관찰)
  vowel_substitution      단모음 대치      ㅗ↔ㅜ 등                    (사례 관찰)
  sibilant_confusion      ㅅ↔ㅈ/ㅊ 혼동   자두→사두, 진달래→신달래     (사례 관찰)
  phonation_confusion     평/경/격음 혼동  ㄲ↔ㄸ 눈꺼풀→눈떠풀          (사례 관찰)
  place_confusion         조음위치 혼동    배↔대(ㅂ↔ㄷ), 뷔↔귀         (사례 관찰)
  manner_confusion        조음방법 혼동    그 외 자음 대치
  rule_failure:<규칙>     음운변동 미적용  국물→국물(비음화 실패)        (사례 관찰)
  rule_overapplication    음운변동 과적용

각 오류유형 → 약점 하위기술(skill_map 노드) + 심각도를 함께 반환해
배치/처방 엔진(skill_map.py)과 종단 프로파일(learner_profile.py)이 바로 소비한다.
"""

from collections import namedtuple

from align import Jamo, decompose, diff_tokens

# ── 임상 오류 1건 ──
#   pattern  : 위 카테고리 문자열
#   role     : onset|nucleus|coda|word(도치 등 음절경계 오류)
#   skill    : 이 오류가 가리키는 약점 하위기술(skill_map 노드 id)
#   severity : 'high'|'med'|'low' (난독증 변별 가중)
#   detail   : 사람용 설명(예 'ㅅ→ㅈ')
#   exp/act  : 한글 자모(표시용)
ClinicalError = namedtuple(
    "ClinicalError", ["pattern", "role", "skill", "severity", "detail", "exp", "act"]
)

# ──────────────────────────────────────────────────────────────────────────
# 조음 자질표 (한국어 자음)
# ──────────────────────────────────────────────────────────────────────────
# place: 조음위치, manner: 조음방법, phon: 발성유형(평/경/격), sib: 치찰성
_PLACE = {
    "ㅂ": "labial", "ㅃ": "labial", "ㅍ": "labial", "ㅁ": "labial",
    "ㄷ": "alveolar", "ㄸ": "alveolar", "ㅌ": "alveolar", "ㄴ": "alveolar",
    "ㄹ": "alveolar", "ㅅ": "alveolar", "ㅆ": "alveolar",
    "ㅈ": "palatal", "ㅉ": "palatal", "ㅊ": "palatal",
    "ㄱ": "velar", "ㄲ": "velar", "ㅋ": "velar", "ㅇ": "velar",
    "ㅎ": "glottal",
}
_MANNER = {
    "ㅂ": "stop", "ㅃ": "stop", "ㅍ": "stop", "ㄷ": "stop", "ㄸ": "stop",
    "ㅌ": "stop", "ㄱ": "stop", "ㄲ": "stop", "ㅋ": "stop",
    "ㅁ": "nasal", "ㄴ": "nasal", "ㅇ": "nasal",
    "ㄹ": "liquid",
    "ㅅ": "fricative", "ㅆ": "fricative", "ㅎ": "fricative",
    "ㅈ": "affricate", "ㅉ": "affricate", "ㅊ": "affricate",
}
_PHON = {  # 평음(lax)/경음(tense)/격음(aspirate)
    "ㄱ": "lax", "ㄷ": "lax", "ㅂ": "lax", "ㅅ": "lax", "ㅈ": "lax",
    "ㄲ": "tense", "ㄸ": "tense", "ㅃ": "tense", "ㅆ": "tense", "ㅉ": "tense",
    "ㅋ": "asp", "ㅌ": "asp", "ㅍ": "asp", "ㅊ": "asp", "ㅎ": "asp",
}
# 치찰음 계열(ㅅ↔ㅈ↔ㅊ 혼동 = 사례연구 최다 청각변별 오류, 사례 관찰)
_SIBILANT = set("ㅅㅆㅈㅉㅊ")

# ──────────────────────────────────────────────────────────────────────────
# 모음 자질표 (활음 + 기본모음)
# ──────────────────────────────────────────────────────────────────────────
# (glide, base): glide ∈ {none, y, w}, base = 활음 제거한 단모음
_VOWEL = {
    "ㅏ": ("none", "ㅏ"), "ㅐ": ("none", "ㅐ"), "ㅓ": ("none", "ㅓ"),
    "ㅔ": ("none", "ㅔ"), "ㅗ": ("none", "ㅗ"), "ㅜ": ("none", "ㅜ"),
    "ㅡ": ("none", "ㅡ"), "ㅣ": ("none", "ㅣ"),
    # y계(미끄러지는 모음, 사례 관찰)
    "ㅑ": ("y", "ㅏ"), "ㅒ": ("y", "ㅐ"), "ㅕ": ("y", "ㅓ"),
    "ㅖ": ("y", "ㅔ"), "ㅛ": ("y", "ㅗ"), "ㅠ": ("y", "ㅜ"),
    # w계(이중모음, 사례 관찰)
    "ㅘ": ("w", "ㅏ"), "ㅙ": ("w", "ㅐ"), "ㅚ": ("w", "ㅔ"),
    "ㅝ": ("w", "ㅓ"), "ㅞ": ("w", "ㅔ"), "ㅟ": ("w", "ㅣ"),
    "ㅢ": ("w", "ㅣ"),  # ㅢ는 별도지만 활음형으로 취급
}


def _feat(ch):
    return {
        "place": _PLACE.get(ch), "manner": _MANNER.get(ch),
        "phon": _PHON.get(ch), "sib": ch in _SIBILANT,
    }


def _classify_onset_sub(exp_ch, act_ch):
    """초성↔초성 대치 → (pattern, skill, severity, detail).

    핵심: 초성 'ㅇ'은 음가 없는 무음(null). 자음→ㅇ 대치는 실제로 '초성 생략'이고
    ㅇ→자음은 '초성 첨가'다 (가죽→아욱 처럼 difflib이 onset 자리 대치로 보는 경우).
    """
    # 무음 초성 처리: 자음→ㅇ = 초성 생략(난독증 핵심 신호)
    if act_ch == "ㅇ" and exp_ch != "ㅇ":
        return "onset_deletion", "pa_phoneme", "high", f"{exp_ch}→∅"
    if exp_ch == "ㅇ" and act_ch != "ㅇ":
        return "onset_insertion", "pa_phoneme", "med", f"∅→{act_ch}"
    fe, fa = _feat(exp_ch), _feat(act_ch)
    detail = f"{exp_ch}→{act_ch}"
    # 1) 치찰음 혼동 (ㅅ↔ㅈ↔ㅊ) — 사례 최다, 청각변별 핵심
    if fe["sib"] and fa["sib"] and exp_ch != act_ch:
        return "sibilant_confusion", "percept_sibilant", "high", detail
    # 2) 발성유형(평/경/격) 혼동 — 위치·방법 같고 phon만 다름
    if fe["place"] == fa["place"] and fe["manner"] == fa["manner"] \
            and fe["phon"] != fa["phon"]:
        return "phonation_confusion", "percept_consonant", "med", detail
    # 3) 조음위치 혼동 — 방법 같고 위치만 다름 (배↔대)
    if fe["manner"] == fa["manner"] and fe["place"] != fa["place"]:
        return "place_confusion", "percept_consonant", "med", detail
    # 4) 그 외 자음 대치 — 조음방법 혼동
    return "manner_confusion", "percept_consonant", "med", detail


def _classify_vowel_sub(exp_ch, act_ch):
    """중성↔중성 대치 → (pattern, skill, severity, detail)."""
    ge, be = _VOWEL.get(exp_ch, ("none", exp_ch))
    ga, ba = _VOWEL.get(act_ch, ("none", act_ch))
    detail = f"{exp_ch}→{act_ch}"
    # 1) 활음 단순화: 기대는 활음 있는데 실제는 활음 없음 (야→아, 사례 관찰)
    if ge != "none" and ga == "none":
        skill = "decode_vowel_glide" if ge == "y" else "decode_vowel_diphthong"
        return "glide_simplification", skill, "high", detail
    # 2) 활음 첨가: 기대는 단모음인데 활음을 넣음 (카→콰, 키→퀴, 사례 관찰)
    if ge == "none" and ga != "none":
        return "glide_insertion", "decode_vowel_diphthong", "med", detail
    # 3) w계 이중모음끼리 혼동 (웨↔워↔왜, 사례 관찰)
    if ge == "w" and ga == "w":
        return "w_diphthong_confusion", "decode_vowel_diphthong", "high", detail
    # 4) y계끼리 혼동
    if ge == "y" and ga == "y":
        return "glide_simplification", "decode_vowel_glide", "med", detail
    # 5) 단모음 대치 (ㅗ↔ㅜ 등, 사례 관찰)
    return "vowel_substitution", "decode_vowel_simple", "med", detail


def _classify_coda_sub(exp_ch, act_ch):
    detail = f"{exp_ch}(받침)→{act_ch}(받침)"
    return "coda_substitution", "rule_final7", "med", detail


def _classify_single_sub(exp_jamo, act_jamo):
    """단일 자모 대치(같은 role) 분류."""
    role = exp_jamo.role
    e, a = exp_jamo.char, act_jamo.char
    if role == "onset":
        return _classify_onset_sub(e, a)
    if role == "nucleus":
        return _classify_vowel_sub(e, a)
    if role == "coda":
        return _classify_coda_sub(e, a)
    return "other_substitution", "decode_onset", "low", f"{e}→{a}"


def _is_transposition(exp_tokens, act_tokens):
    """음절/자모 도치 여부: 같은 자모 구성인데 순서만 바뀜 (가방→바강, 사례 관찰).

    조건: 자모 멀티셋(char,role)이 동일하지만 순서(char 시퀀스)가 다름.
    """
    if len(exp_tokens) != len(act_tokens) or len(exp_tokens) < 4:
        return False
    ek = sorted((t.char, t.role) for t in exp_tokens)
    ak = sorted((t.char, t.role) for t in act_tokens)
    eseq = [t.char for t in exp_tokens]
    aseq = [t.char for t in act_tokens]
    return ek == ak and eseq != aseq


def classify_errors(errors):
    """raw Error 리스트(같은 단어) → ClinicalError 리스트.

    대치는 자질로 세분, 생략/첨가는 role로 분류. 도치는 classify_word에서 처리.
    """
    out = []
    for e in errors:
        if e.kind == "substitution" and len(e.exp) == 1 and len(e.act) == 1:
            pat, skill, sev, det = _classify_single_sub(e.exp[0], e.act[0])
            out.append(ClinicalError(pat, e.exp[0].role, skill, sev, det,
                                     e.exp, e.act))
        elif e.kind == "substitution":
            # 다중 자모 대치(정렬 모호) → 각 쌍 근사 분류, role은 첫 토큰 기준
            role = e.exp[0].role if e.exp else "onset"
            det = "".join(t.char for t in e.exp) + "→" + "".join(t.char for t in e.act)
            out.append(ClinicalError("cluster_substitution", role,
                                     "decode_blend", "med", det, e.exp, e.act))
        elif e.kind == "omission":
            role = e.exp[0].role if e.exp else "onset"
            det = "".join(t.char for t in e.exp) + "→∅"
            if role == "onset":
                out.append(ClinicalError("onset_deletion", "onset",
                                         "pa_phoneme", "high", det, e.exp, []))
            elif role == "coda":
                out.append(ClinicalError("coda_deletion", "coda",
                                         "decode_coda", "low", det, e.exp, []))
            elif role == "nucleus":
                out.append(ClinicalError("nucleus_deletion", "nucleus",
                                         "decode_vowel_simple", "high", det, e.exp, []))
            else:
                out.append(ClinicalError("deletion", role, "decode_onset",
                                         "med", det, e.exp, []))
        elif e.kind == "insertion":
            role = e.act[0].role if e.act else "onset"
            det = "∅→" + "".join(t.char for t in e.act)
            out.append(ClinicalError("insertion", role, "pa_phoneme",
                                     "med", det, [], e.act))
    return out


def classify_word(expected_pron, actual_text, errors=None):
    """단어 1개의 임상 오류 프로파일.

    expected_pron : 기대 발음(G2P 결과, 한글)
    actual_text   : 실제 발화(한글; ASR 또는 dry-run)
    errors        : 이미 추출한 raw Error 리스트(없으면 여기서 diff)

    반환: ClinicalError 리스트. 음절 도치는 단일 word-level 오류로 우선 처리.
    """
    exp_tokens = decompose(expected_pron)
    act_tokens = decompose(actual_text)
    if _is_transposition(exp_tokens, act_tokens):
        det = f"{expected_pron}→{actual_text}"
        return [ClinicalError("syllable_transposition", "word", "pa_syllable",
                              "high", det, exp_tokens, act_tokens)]
    if errors is None:
        errors = diff_tokens(exp_tokens, act_tokens)
    return classify_errors(errors)


def classify_rule_failure(rule, applied_correctly):
    """음운변동 문항 전용: 규칙 적용 여부 → rule_failure 오류 1건.

    rule              : 규칙명(연음/비음화/경음화/구개음화/격음화/유음화/ㅎ탈락/겹받침/사이시옷)
    applied_correctly : 규칙대로 읽었는가(True면 오류 없음)
    """
    if applied_correctly:
        return None
    skill = "rule_" + _RULE_KEY.get(rule, "etc")
    return ClinicalError(f"rule_failure:{rule}", "word", skill, "high",
                         f"{rule} 미적용", [], [])


# 규칙명 → skill_map 노드 접미사
_RULE_KEY = {
    "연음": "liaison", "비음화": "nasal", "경음화": "tense", "구개음화": "palatal",
    "격음화(축약)": "aspirate", "격음화": "aspirate", "유음화": "lateral",
    "설측음화": "lateral", "ㅎ탈락": "hdrop", "겹받침": "cluster",
    "겹받침+경음화": "cluster", "사이시옷": "saisiot",
}


def summarize_profile(clinical_errors):
    """ClinicalError 리스트 → 오류유형별 집계 + 약점 스킬 + 가중 심각도.

    반환 dict:
      patterns : {pattern: count}
      skills   : {skill: weighted_score}  (severity 가중 누적; 클수록 약점)
      top      : [(skill, score)] 내림차순
    """
    _W = {"high": 3.0, "med": 1.5, "low": 0.5}
    patterns, skills = {}, {}
    for ce in clinical_errors:
        patterns[ce.pattern] = patterns.get(ce.pattern, 0) + 1
        skills[ce.skill] = skills.get(ce.skill, 0.0) + _W.get(ce.severity, 1.0)
    top = sorted(skills.items(), key=lambda kv: -kv[1])
    return {"patterns": patterns, "skills": skills, "top": top}


if __name__ == "__main__":
    # PPT 실관찰 오류로 분류 시연
    cases = [
        ("가죽", "아욱", "초성 생략(사례 관찰)"),
        ("가방", "바강", "음절 도치(사례 관찰)"),
        ("자두", "사두", "ㅈ→ㅅ 치찰음(사례 관찰)"),
        ("진달래", "신달래", "ㅈ→ㅅ 치찰음(사례 관찰)"),
        ("카", "콰", "활음 첨가(사례 관찰)"),
        ("야", "아", "활음 단순화(사례 관찰)"),
        ("웨이터", "워이터", "w계 모음 혼동(사례 관찰)"),
        ("눈꺼풀", "눈떠풀", "ㄲ→ㄸ 발성/위치(사례 관찰)"),
        ("멀튼", "머튼", "종성 생략(사례 관찰)"),
    ]
    for exp, act, note in cases:
        ces = classify_word(exp, act)
        tags = ", ".join(f"{c.pattern}[{c.severity}]({c.detail})" for c in ces)
        print(f"{exp}→{act:5} {tags:50} # {note}")
    print()
    allce = [c for exp, act, _ in cases for c in classify_word(exp, act)]
    s = summarize_profile(allce)
    print("약점 스킬(가중):", s["top"])
