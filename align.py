"""자모 분해 + 정렬 + 오류 분류.

기대 발음과 실제 발화를 자모 단위로 비교해 대치/생략/첨가를 뽑는다.

굳이 role 태그를 붙여 분해하는 이유: 단순 j2hcj(h2j()) 분해는 위치를 잃어
초성 ㄱ과 종성 ㄱ을 못 가린다. 그런데 transparent 채점("종성 조음은 관대,
단 다른 단어가 되면 오류")은 '이게 종성이냐'를 알아야 한다.

Jamo = (char, role, syl): 자모 문자, onset/nucleus/coda, 원래 음절 인덱스.
"""

from collections import namedtuple
from difflib import SequenceMatcher

Jamo = namedtuple("Jamo", ["char", "role", "syl"])

# 표준 한글 분해 테이블 (유니코드 음절 = 0xAC00 + (cho*21 + jung)*28 + jong)
_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

_SBASE, _SLAST = 0xAC00, 0xD7A3


def decompose(text):
    """텍스트 → 위치 태그가 붙은 Jamo 토큰 리스트.

    한글 음절은 초/중/종으로 분해, 그 외 문자(공백 제외)는 role='other'로 통과.
    겹받침(예 ㄳ)은 한 토큰으로 둔다(발음 단위 비교에는 충분; 필요시 v1에서 세분).
    """
    out = []
    for syl_idx, ch in enumerate(text):
        if ch.isspace():
            continue
        code = ord(ch)
        if _SBASE <= code <= _SLAST:
            s = code - _SBASE
            cho, jung, jong = s // 588, (s % 588) // 28, s % 28
            out.append(Jamo(_CHO[cho], "onset", syl_idx))
            out.append(Jamo(_JUNG[jung], "nucleus", syl_idx))
            if jong:
                out.append(Jamo(_JONG[jong], "coda", syl_idx))
        else:
            out.append(Jamo(ch, "other", syl_idx))
    return out


# 오류 1건: kind=substitution|omission|insertion,
#   exp/act = Jamo 리스트(해당 구간), 비교 편의를 위해 원본 토큰 보존.
Error = namedtuple("Error", ["kind", "exp", "act"])


def _key_jamo(j):
    """자모 모드 정렬 키: 같은 'ㄱ'이라도 초성↔종성이면 다른 토큰."""
    return (j.char, j.role)


def _key_char(j):
    """음소 모드 정렬 키: 기호만 비교(출력에 role 없음)."""
    return j.char


def diff_tokens(exp, act, key=_key_jamo):
    """두 Jamo 토큰 리스트를 정렬해 Error 리스트 반환(코어)."""
    exp_key = [key(j) for j in exp]
    act_key = [key(j) for j in act]
    errors = []
    sm = SequenceMatcher(None, exp_key, act_key, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            errors.append(Error("substitution", exp[i1:i2], act[j1:j2]))
        elif op == "delete":
            errors.append(Error("omission", exp[i1:i2], []))
        elif op == "insert":
            errors.append(Error("insertion", [], act[j1:j2]))
    return errors


def diff_errors(expected_text, actual_text):
    """기대 발음 vs 실제 발화 → 오류(Error) 리스트."""
    return diff_tokens(decompose(expected_text), decompose(actual_text))


def _editcost(exp, act, key=_key_jamo):
    """Jamo 토큰 리스트 간 편집거리(정렬 비용). key 일치=0."""
    a = [key(j) for j in exp]
    b = [key(j) for j in act]
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
        prev = cur
    return prev[m]


def _seg_cap(word_len):
    """단어에 배정 가능한 실제 세그먼트 최대 길이.

    낱말/어절 읽기에서 한 단어의 실제 발화는 기대 음소열 길이 근처다(대치·삽입
    으로 몇 개 늘 수 있음). 무한정 긴 구간을 후보로 두면 DP가 O(N·M²·L)로
    폭증하므로 '단어 길이에 비례하는 밴드'로 세그먼트 길이를 제한한다."""
    return 2 * word_len + 4


def align_tokens_dp(exp_words, act, key=_key_jamo):
    """DP 정렬: 실제 토큰열 act를 단어 순서대로 '연속 구간'으로 최적 분할해
    각 기대 단어(exp_words[i] 토큰열)에 배정. 누락 단어는 빈 구간.

    exp_words : list[list[Jamo]]  단어별 기대 토큰
    act       : list[Jamo]        실제 발화 토큰(연결)
    반환: 단어별 배정 토큰 리스트(exp_words와 같은 길이).

    세그먼트 길이를 _seg_cap()으로 제한(밴드)해 복잡도를 O(N·M·B)로 낮춘다.
    B = 최대 세그먼트 길이. 여러 단어를 한 번에 정렬할 때도 실사용 속도가 난다.
    """
    N, M = len(exp_words), len(act)
    INF = float("inf")
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    back = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(1, N + 1):
        w = exp_words[i - 1]
        cap = _seg_cap(len(w))
        for p in range(M + 1):
            best, bq = INF, 0
            # 세그먼트 길이(p-q)를 cap 이내로만 탐색 → 밴드 DP
            for q in range(max(0, p - cap), p + 1):
                if dp[i - 1][q] == INF:
                    continue
                c = dp[i - 1][q] + _editcost(w, act[q:p], key)
                if c < best:
                    best, bq = c, q
            dp[i][p] = best
            back[i][p] = bq
    # 밴드로 인해 dp[N][M]이 도달 불가할 수 있으므로, 마지막 행에서 최소비용 p로 종료
    p = min(range(M + 1), key=lambda x: dp[N][x])
    tail = act[p:]  # 밴드 밖으로 남은 실제 토큰(마지막 단어에 흡수)
    segs = [None] * N
    for i in range(N, 0, -1):
        q = back[i][p]
        segs[i - 1] = act[q:p]
        p = q
    if tail and segs:
        segs[-1] = segs[-1] + tail
    return segs


def align_words_dp(expected_prons, actual_text):
    """자모 모드 목록 정렬(하위호환): 한글 발음열 기준."""
    exp_words = [decompose(p) for p in expected_prons]
    return align_tokens_dp(exp_words, decompose(actual_text), _key_jamo)


def diff_errors_by_word(expected_prons, actual_text):
    """목록 읽기용(자모): DP 정렬 후 단어별 Error 리스트 반환."""
    exp_words = [decompose(p) for p in expected_prons]
    segs = align_words_dp(expected_prons, actual_text)
    return [diff_tokens(exp_words[i], segs[i]) for i in range(len(expected_prons))]


def diff_errors_by_word_tokens(exp_word_tokens, act_tokens, key=_key_jamo):
    """목록 읽기용(일반): 토큰열을 직접 받아 DP 정렬 후 단어별 Error 리스트.
    음소 모드는 exp/act 음소 토큰 + key=_key_char 로 호출."""
    segs = align_tokens_dp(exp_word_tokens, act_tokens, key)
    return [diff_tokens(exp_word_tokens[i], segs[i], key) for i in range(len(exp_word_tokens))]


def jamo_str(tokens):
    """디버그 출력용."""
    return " ".join(f"{t.char}" for t in tokens)


_JONG_REV = {c: i for i, c in enumerate(_JONG)}
_CHO_REV = {c: i for i, c in enumerate(_CHO)}
_JUNG_REV = {c: i for i, c in enumerate(_JUNG)}


def tokens_to_text(tokens):
    """role 태그가 있는 Jamo 토큰열 → 한글 문자열로 재조합(들린대로 표시용)."""
    out, cho, jung = [], None, None

    def flush(coda=""):
        nonlocal cho, jung
        if jung is not None:
            ci = _CHO_REV.get(cho or "ㅇ", 11)
            ji = _JUNG_REV.get(jung)
            ki = _JONG_REV.get(coda, 0) if coda else 0
            if ji is not None:
                out.append(chr(0xAC00 + (ci * 21 + ji) * 28 + ki))
            else:
                out.append((cho or "") + jung + (coda or ""))
        elif cho:
            out.append(cho)
        cho, jung = None, None

    for t in tokens:
        if t.role == "onset":
            flush()
            cho = t.char
        elif t.role == "nucleus":
            if jung is not None:
                flush()
            jung = t.char
        elif t.role == "coda":
            flush(t.char)
        else:
            flush()
            out.append(t.char)
    flush()
    return "".join(out)


if __name__ == "__main__":
    cases = [
        ("책상", "채상"),   # 종성 ㄱ 생략
        ("나비", "나무"),   # 대치
        ("학교", "학꾜야"),  # 첨가
        ("궁물", "궁물"),   # 동일
    ]
    for exp, act in cases:
        errs = diff_errors(exp, act)
        print(f"{exp} vs {act}:")
        for e in errs:
            print(f"   {e.kind:12} exp={jamo_str(e.exp):8} act={jamo_str(e.act)}")
        if not errs:
            print("   (no error)")
