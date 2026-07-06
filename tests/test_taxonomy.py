"""임상 계층 검증 — 관찰 오류쌍을 ground truth로.

익명화된 임상 사례연구에 기록된 관찰 오류를 분류기가 맞는 임상유형으로 잡는지 본다.
ASR 없이 (expected→actual) 문자열 주입. pytest 또는 python tests/test_taxonomy.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_taxonomy import classify_word, summarize_profile  # noqa: E402
from skill_map import placement, prescribe, SKILLS  # noqa: E402
from phon_awareness import make_task, score_response  # noqa: E402


def _patterns(exp, act):
    return [c.pattern for c in classify_word(exp, act)]


# ── error_taxonomy: 사례연구 실관찰 오류 ──
def test_onset_deletion():            # 사례 관찰: 가죽→아욱
    assert "onset_deletion" in _patterns("가죽", "아죽")
    assert "onset_deletion" in _patterns("까마귀", "아마귀")


def test_syllable_transposition():    # 사례 관찰: 가방→바강
    assert _patterns("가방", "바강") == ["syllable_transposition"]


def test_sibilant_confusion():        # 사례 관찰: 자두→사두, 진달래→신달래
    assert "sibilant_confusion" in _patterns("자두", "사두")
    assert "sibilant_confusion" in _patterns("진달래", "신달래")
    assert "sibilant_confusion" in _patterns("삼촌", "잠촌")


def test_glide_simplification():      # 사례 관찰: 야→아 (활음 단순화)
    assert "glide_simplification" in _patterns("야구", "아구")


def test_glide_insertion():           # 사례 관찰: 카→콰, 키→퀴 (활음 첨가)
    assert "glide_insertion" in _patterns("카", "콰")


def test_w_diphthong_confusion():     # 사례 관찰: 웨이터→워이터
    assert "w_diphthong_confusion" in _patterns("웨이터", "워이터")


def test_phonation_or_place_confusion():  # 사례 관찰: 눈꺼풀→눈떠풀 (ㄲ→ㄸ)
    pats = _patterns("눈꺼풀", "눈떠풀")
    assert "place_confusion" in pats or "phonation_confusion" in pats


def test_coda_deletion():             # 사례 관찰: 멀튼→머튼 (받침 ㄹ 생략)
    assert "coda_deletion" in _patterns("멀튼", "머튼")


def test_correct_no_error():
    assert classify_word("바다", "바다") == []


# ── summarize_profile: 약점 스킬 집계 ──
def test_summary_weights_sibilant_high():
    # 치찰음(high) 오류가 단모음 대치(med)보다 가중↑
    ces = classify_word("자두", "사두") + classify_word("거미", "고미")
    s = summarize_profile(ces)
    assert s["skills"]["percept_sibilant"] > s["skills"]["decode_vowel_simple"]


# ── skill_map: 배치/처방 ──
def test_placement_current_is_lowest_unmastered():
    ev = {"decode_vowel_simple": 0.95, "decode_vowel_glide": 0.4}
    pl = placement(ev)
    assert pl.current == "decode_vowel_glide"  # 단모음 숙달, 미끄러지는모음 막힘


def test_placement_prereq_gating_next_targets():
    # 단모음 미숙달이면 미끄러지는모음은 다음목표 후보 아님(전제조건 미충족)
    ev = {"decode_vowel_simple": 0.5}
    pl = placement(ev)
    assert "decode_vowel_glide" not in pl.next_targets


def test_prescribe_orders_by_developmental_level():
    weak = [("decode_vowel_glide", 6.0), ("percept_sibilant", 3.0)]
    rx = prescribe(weak, {})
    # 낮은 level(치찰음 변별 level14) 처방이 미끄러지는모음(level16)보다 먼저
    ids = [p["skill"] for p in rx]
    assert ids.index("percept_sibilant") < ids.index("decode_vowel_glide")


def test_all_taxonomy_skills_exist_in_map():
    # 분류기가 가리키는 모든 skill 노드가 skill_map에 정의돼 있어야(처방 가능)
    sample_pairs = [("가죽", "아죽"), ("가방", "바강"), ("자두", "사두"),
                    ("야구", "아구"), ("카", "콰"), ("웨이터", "워이터"),
                    ("멀튼", "머튼"), ("국물", "국물")]
    for exp, act in sample_pairs:
        for ce in classify_word(exp, act):
            assert ce.skill in SKILLS, f"{ce.skill} not in skill_map ({exp}→{act})"


# ── phon_awareness: 음운인식 과제 생성/채점 ──
def test_pa_syllable_reverse():       # 사례 관찰: 음절 도치
    t = make_task("syllable", "reverse", "나무")
    assert t.expected == "무나"
    assert score_response(t, "무나")["correct"]
    assert not score_response(t, "나무")["correct"]  # 도치 실패


def test_pa_syllable_delete():        # 사례 관찰: 축구-구→축
    t = make_task("syllable", "delete", "축구", target="구")
    assert t.expected == "축"
    assert score_response(t, "축")["correct"]


def test_pa_phoneme_blend():          # 사례 관찰: ㅂ+ㅓ→버
    t = make_task("phoneme", "blend", "", jamos=["ㅂ", "ㅓ"])
    assert t.expected == "버"


def test_pa_phoneme_blend_coda():     # ㄱ+ㅏ+ㅇ→강
    t = make_task("phoneme", "blend", "", jamos=["ㄱ", "ㅏ", "ㅇ"])
    assert t.expected == "강"


def test_pa_phoneme_delete_onset():   # 사례 관찰: 기-ㄱ→이
    t = make_task("phoneme", "delete", "기")
    assert t.expected == "이"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
