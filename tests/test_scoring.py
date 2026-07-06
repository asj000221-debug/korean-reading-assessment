"""문항 유형별 채점 룰 단위 테스트.

actual_text를 직접 주입해 ASR 없이 scoring 로직만 본다.
pytest로도, python tests/test_scoring.py로도 돈다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import score_word  # noqa: E402


# ---------- nonword: 가장 엄격 ----------
def test_nonword_correct():
    s = score_word("낟", "nonword", expected_pron="낟", actual_text="낟")
    assert s.correct and s.accuracy == 1.0


def test_nonword_coda_sub_is_error():
    # 종성 ㄷ→ㅅ: 무의미단어는 종성도 엄격히 오류
    s = score_word("낟", "nonword", expected_pron="낟", actual_text="낫")
    assert not s.correct
    assert any(e.weight > 0 for e in s.errors)


def test_nonword_omission_is_error():
    # 멀튼 → 머튼 (ㄹ 종성 생략)
    s = score_word("멀튼", "nonword", expected_pron="멀튼", actual_text="머튼")
    assert not s.correct


# ---------- phonrule: 음운규칙 적용 실패 = 오류 ----------
def test_phonrule_applied_correct():
    # 국물을 규칙대로 [궁물]로 읽음
    s = score_word("국물", "phonrule", expected_pron="궁물", actual_text="궁물")
    assert s.correct and s.accuracy == 1.0


def test_phonrule_failure_is_error():
    # 비음화 미적용: [국물]로 읽음 → 오류로 잡혀야 함
    s = score_word("국물", "phonrule", expected_pron="궁물", actual_text="국물")
    assert not s.correct
    assert any(e.tag == "phonrule_failure" for e in s.errors)


def test_phonrule_yeoneum_failure():
    # 꽃을 → [꼬츨] 기대. [꼬슬]로 읽으면 오류
    s = score_word("꽃을", "phonrule", expected_pron="꼬츨", actual_text="꼬슬")
    assert not s.correct

# ---------- transparent: 종성 조음 관대, 단 다른 단어 되면 오류 ----------
def test_transparent_correct():
    s = score_word("바다", "transparent", expected_pron="바다", actual_text="바다")
    assert s.correct and s.accuracy == 1.0


def test_transparent_coda_waived():
    # 강 → 가 (종성 ㅇ 생략). 실단어 사전 없으면 관대 처리 → 정답
    s = score_word("강", "transparent", expected_pron="강", actual_text="가")
    assert s.correct
    assert any(e.tag == "coda_lenient_waived" and e.weight == 0.0 for e in s.errors)


def test_transparent_coda_becomes_real_word_is_error():
    # 강 → 가, 그런데 '가'가 실단어라 다른 단어가 됨 → 오류로 카운트
    s = score_word(
        "강", "transparent", expected_pron="강", actual_text="가", real_words={"가"}
    )
    assert not s.correct
    assert any(e.tag == "transparent_real_word" for e in s.errors)


def test_transparent_nucleus_error_not_waived():
    # 중성 대치(바다→바도)는 종성 관대 대상이 아님 → 오류
    s = score_word("바다", "transparent", expected_pron="바다", actual_text="바도")
    assert not s.correct


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
