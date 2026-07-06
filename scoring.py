"""문항 유형별 채점 룰. 핵심 로직이지만 전부 조건문이다(학습 없음).

align이 뽑은 raw 오류 목록에 유형별 규칙을 얹어 최종 점수를 낸다.
- nonword: 모든 자모 차이를 오류로(가장 엄격). 맥락 추측 불가 → 순수 해독력
- phonrule: 음운규칙 적용 실패도 오류로 카운트(이게 측정 대상)
- transparent: 종성 단순 조음 오류는 감점 제외(가중치 0). 단, 받침이 바뀌어
  '다른 실단어'가 되면 오류로 친다.

WordScore로 정/오·오류 태그·정확도(0~1)를 돌려준다.
"""

from collections import namedtuple

from align import diff_errors

# 채점된 개별 오류: error(Error), weight(감점 가중치), tag(설명)
ScoredError = namedtuple("ScoredError", ["kind", "weight", "tag", "exp", "act"])

WordScore = namedtuple(
    "WordScore",
    ["word", "item_type", "correct", "accuracy", "errors", "n_jamo"],
)

# 정/오 판정 임계값(가중 오류=0 이면 정답). transparent는 관대 오류로 0이 될 수 있음.
CORRECT_THRESHOLD = 1.0  # accuracy >= 1.0 이면 correct


def _char_count(tokens):
    """오류 구간의 자모 개수(가중치 기본 단위)."""
    return max(len(tokens), 1)  # insert/delete 한 쪽이 비어도 최소 1


def _is_coda_only(error):
    """이 오류가 '종성(받침)에 한정된' 오류인지.

    transparent 관대 처리 대상: 종성 조음 미숙(받침 생략/약화/대치)이며
    초성·중성은 건드리지 않은 경우. 음소 모드에서 실제측 role 미상('')은 무시.
    """
    roles = {t.role for t in error.exp}
    roles |= {t.role for t in error.act if t.role}  # 미상('')은 제외
    return bool(roles) and roles <= {"coda"}


def score_word(word, item_type, expected_pron, actual_text, real_words=None):
    """단어 1개 채점.

    word         : 원본 철자(리포트/사전조회용)
    item_type    : 'nonword' | 'phonrule' | 'transparent'
    expected_pron: 기대 발음(보통 g2p_expected.expected_prons()[0])
    actual_text  : ASR 인식 결과
    real_words   : transparent에서 '다른 단어가 됨'을 판정할 실단어 집합(선택).
                   actual_text가 여기 있고 expected와 다르면 종성 관대 처리를 끈다.
    """
    if real_words is None:
        real_words = set()

    raw_errors = diff_errors(expected_pron, actual_text)
    # transparent에서 '실단어로 바뀌었나' (종성 관대 처리 해제 조건)
    became_real_word = (
        item_type == "transparent"
        and actual_text != expected_pron
        and actual_text in real_words
    )
    return score_from_errors(
        word, item_type, expected_pron, raw_errors, became_real_word
    )


def score_from_errors(word, item_type, expected_pron, raw_errors,
                      became_real_word=False, n_units=None):
    """이미 추출된 오류 목록으로 채점(단어/목록/음소 모드 공유 코어).

    n_units: 정확도 분모(기대 단위 수). None이면 한글 자모 수로 계산.
             음소 모드는 음소 개수를 직접 전달한다.
    """
    if n_units is None:
        from align import decompose
        n_units = len(decompose(expected_pron))
    n_jamo = max(n_units, 1)

    scored = []
    for e in raw_errors:
        n = _char_count(e.exp if e.kind != "insertion" else e.act)

        if item_type == "nonword":
            # 가장 엄격: 모든 차이 = 오류
            scored.append(ScoredError(e.kind, float(n), "nonword_strict", e.exp, e.act))

        elif item_type == "phonrule":
            # 음운규칙 적용 실패 포함 모든 차이 = 오류
            scored.append(
                ScoredError(e.kind, float(n), "phonrule_failure", e.exp, e.act)
            )

        elif item_type == "transparent":
            if _is_coda_only(e) and not became_real_word:
                # 종성 단순 조음 오류 → 감점 제외(가중치 0), 태그만 남김
                scored.append(
                    ScoredError(e.kind, 0.0, "coda_lenient_waived", e.exp, e.act)
                )
            else:
                tag = "transparent_real_word" if became_real_word else "transparent_error"
                scored.append(ScoredError(e.kind, float(n), tag, e.exp, e.act))
        else:
            raise ValueError(f"unknown item_type: {item_type!r}")

    weighted = sum(s.weight for s in scored)
    accuracy = max(0.0, 1.0 - weighted / n_jamo)
    correct = accuracy >= CORRECT_THRESHOLD

    return WordScore(word, item_type, correct, round(accuracy, 4), scored, n_jamo)


def format_report(scores):
    """단어별/전체 정확도 리포트 문자열."""
    lines = []
    n_correct = sum(1 for s in scores if s.correct)
    total = len(scores)
    for s in scores:
        mark = "O" if s.correct else "X"
        lines.append(f"[{mark}] {s.word:8} ({s.item_type:11}) acc={s.accuracy:.2f}")
        for e in s.errors:
            exp = "".join(t.char for t in e.exp) or "-"
            act = "".join(t.char for t in e.act) or "-"
            lines.append(f"       {e.kind:12} w={e.weight:.1f} [{e.tag}] {exp}->{act}")
    overall = n_correct / total if total else 0.0
    lines.append(f"\n전체: {n_correct}/{total} 정답  (정확도 {overall:.1%})")
    return "\n".join(lines)
