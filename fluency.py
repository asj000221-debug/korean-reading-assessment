"""STEP 6(부분) — 읽기 유창성: 속도(분당 음절수) 측정.

실제 '읽기 유창성' 검사 = 정확도 + 속도. 정확도는 채점 파이프라인이 내고,
여기서는 속도를 측정한다. 발화 구간(첫~끝 유성 구간, 내부 멈춤 포함)을
읽기 시간으로 보고 분당 음절수(SPM)를 계산.

망설임/자기수정(3초 이상 멈춤, 되돌이) 검출은 음소별 타임스탬프(forced
alignment)가 필요하며 v1 과제로 남긴다. 여기서는 전체 속도만.
"""


def count_syllables(text):
    """한글 음절(가~힣) 개수."""
    return sum(1 for ch in text if "가" <= ch <= "힣")


def fluency_metrics(text, scores, span_sec):
    """text(읽은 문장/목록), scores(어절/단어별 WordScore), span_sec(읽기 시간).

    반환: 전체 음절수, 정확 음절수, 시간, 분당 음절수(SPM), 분당 정확음절수.
    """
    total_syl = count_syllables(text)
    correct_syl = sum(count_syllables(s.word) for s in scores if s.correct)
    minutes = (span_sec / 60.0) if span_sec > 0 else 1e-9
    return {
        "total_syllables": total_syl,
        "correct_syllables": correct_syl,
        "duration_sec": round(span_sec, 2),
        "spm": round(total_syl / minutes, 1),
        "correct_spm": round(correct_syl / minutes, 1),
    }


def format_fluency(m):
    return (f"읽기 시간 {m['duration_sec']}s · 속도 {m['spm']} 음절/분 "
            f"(정확 {m['correct_spm']} 음절/분, {m['correct_syllables']}/{m['total_syllables']} 음절)")


if __name__ == "__main__":
    # 간단 점검
    class S:  # mock WordScore
        def __init__(self, w, c):
            self.word, self.correct = w, c

    sc = [S("해돋이를", True), S("보러", True), S("산에", True),
          S("같이", False), S("올라갔어요", True)]
    print(format_fluency(fluency_metrics("해돋이를 보러 산에 같이 올라갔어요", sc, 3.4)))
