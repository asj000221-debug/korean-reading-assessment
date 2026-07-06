"""음운인식(PA) 과제 — 글자 없는 청각 조작.

기존 시스템은 해독(글자→소리)만 잰다. 그런데 사례의 검사절차는 세 영역 중 첫째가
음운인식이다 — 글자 없이 소리만으로 조작하기(음절/음소 수세기·분절·합성·생략·
첨가·대치·도치).

포인트는 PA 과제의 '기대 응답 = 입력의 변환'이라는 점이다. 아동이 구두로 답하면
기존 ASR+정렬 코어로 기대 응답과 비교해 채점할 수 있다. 여기선 변환 연산을 순수
함수로 두고, 과제 생성(기대 응답 계산)과 채점(응답↔기대 정렬)만 얹는다.
음향 인식은 asr/phoneme_asr 재사용.
"""

from collections import namedtuple

from align import Jamo, decompose, diff_tokens, tokens_to_text

# PA 과제 1건. level=syllable|phoneme, op=count/blend/segment/delete/add/substitute/
# reverse, expected=기대 응답(한글 또는 숫자), skill=pa_syllable|pa_phoneme.
PaTask = namedtuple("PaTask", ["level", "op", "prompt", "stimulus", "expected", "skill"])


# ── 음절 수준 연산 (입력은 한글 음절 문자열) ──
def syl_count(word):
    """음절 수세기: '자동차'→'3'."""
    return str(sum(1 for ch in word if "가" <= ch <= "힣"))


def syl_blend(parts):
    return "".join(parts)


def syl_segment(word):
    """'전화' → '전 화'."""
    return " ".join(ch for ch in word)


def syl_delete(word, target):
    """음절 생략: ('축구','구')→'축'."""
    syls = list(word)
    if target in syls:
        syls.remove(target)
    return "".join(syls)


def syl_add(word, syl, pos="front"):
    return syl + word if pos == "front" else word + syl


def syl_reverse(word):
    """음절 도치: '나무'→'무나'. 도치(가방→바강)는 이 사례의 핵심 약점."""
    return "".join(reversed(list(word)))


def syl_substitute(word, old, new):
    """('호박','호','수') → '수박'."""
    return word.replace(old, new, 1)


# ── 음소 수준 연산 (자모 분해 활용) ──
def _recompose(tokens):
    return tokens_to_text(tokens)


def phon_count(word):
    """음소 수세기(무음 초성 ㅇ 제외): '버'→'2', '강'→'3'."""
    toks = [t for t in decompose(word) if t.char != "ㅇ" or t.role != "onset"]
    return str(len(toks))


def phon_blend(jamos):
    """음소 합성: ['ㅂ','ㅓ']→'버', ['ㄱ','ㅏ','ㅇ']→'강'."""
    return _recompose(_assign_roles(jamos))


def _assign_roles(jamos):
    """자모 나열에 초/중/종 역할 추정(단순: 모음 기준 음절 경계)."""
    _VOW = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
    out, syl = [], 0
    i = 0
    while i < len(jamos):
        j = jamos[i]
        if j in _VOW:
            out.append(Jamo(j, "nucleus", syl))
            # 다음이 자음이고 그 다음이 모음이 아니면 종성
            if i + 1 < len(jamos) and jamos[i + 1] not in _VOW:
                nxt2 = jamos[i + 2] if i + 2 < len(jamos) else None
                if nxt2 is None or nxt2 not in _VOW:
                    out.append(Jamo(jamos[i + 1], "coda", syl))
                    i += 1
            syl += 1
        else:
            out.append(Jamo(j, "onset", syl))
        i += 1
    return out


def phon_delete_onset(word):
    """음소 생략(초성): '기'→'이'(ㄱ 빼기). '가죽'류 초성생략 진단과 짝."""
    toks = decompose(word)
    out, removed = [], False
    for t in toks:
        if not removed and t.role == "onset" and t.char != "ㅇ":
            removed = True
            out.append(Jamo("ㅇ", "onset", t.syl))  # 무음 초성으로
            continue
        out.append(t)
    return _recompose(out)


def phon_substitute_onset(word, new_onset):
    """음소 대치(초성): ('저','ㅊ') → '처'."""
    toks = decompose(word)
    out, done = [], False
    for t in toks:
        if not done and t.role == "onset":
            out.append(Jamo(new_onset, "onset", t.syl))
            done = True
        else:
            out.append(t)
    return _recompose(out)


# ── 과제 생성기 ──
def make_task(level, op, stimulus, **kw):
    """연산 → PaTask(기대 응답 자동 계산)."""
    skill = "pa_syllable" if level == "syllable" else "pa_phoneme"
    if level == "syllable":
        if op == "count":
            exp, pr = syl_count(stimulus), f"'{stimulus}' 몇 음절?"
        elif op == "blend":
            exp, pr = syl_blend(kw["parts"]), f"{'·'.join(kw['parts'])} 합치면?"
        elif op == "segment":
            exp, pr = syl_segment(stimulus), f"'{stimulus}' 나누면?"
        elif op == "delete":
            exp, pr = syl_delete(stimulus, kw["target"]), f"'{stimulus}'에서 '{kw['target']}' 빼면?"
        elif op == "add":
            exp, pr = syl_add(stimulus, kw["syl"], kw.get("pos", "front")), \
                f"'{stimulus}' {kw.get('pos','앞')}에 '{kw['syl']}' 붙이면?"
        elif op == "reverse":
            exp, pr = syl_reverse(stimulus), f"'{stimulus}' 거꾸로 하면?"
        elif op == "substitute":
            exp, pr = syl_substitute(stimulus, kw["old"], kw["new"]), \
                f"'{stimulus}'에서 '{kw['old']}' 대신 '{kw['new']}' 넣으면?"
        else:
            raise ValueError(op)
    else:
        if op == "count":
            exp, pr = phon_count(stimulus), f"'{stimulus}' 소리 몇 개?"
        elif op == "blend":
            exp, pr = phon_blend(kw["jamos"]), f"{'+'.join(kw['jamos'])} 합치면?"
        elif op == "delete":
            exp, pr = phon_delete_onset(stimulus), f"'{stimulus}'에서 첫소리 빼면?"
        elif op == "substitute":
            exp, pr = phon_substitute_onset(stimulus, kw["new_onset"]), \
                f"'{stimulus}' 첫소리를 '{kw['new_onset']}'로 바꾸면?"
        else:
            raise ValueError(op)
    return PaTask(level, op, pr, stimulus, exp, skill)


def score_response(task, response_text):
    """아동 응답(ASR 결과) ↔ 기대 응답 채점.

    반환 dict: correct(bool), accuracy(0~1), expected, response, errors(자모차이).
    count 과제는 숫자 일치로 판정.
    """
    exp, got = task.expected, (response_text or "").strip()
    if task.op == "count":
        # 숫자 또는 한글수사 비교
        got_n = _num(got)
        correct = (got_n == exp) or (got == exp)
        return {"correct": correct, "accuracy": 1.0 if correct else 0.0,
                "expected": exp, "response": got, "errors": []}
    errs = diff_tokens(decompose(exp.replace(" ", "")),
                       decompose(got.replace(" ", "")))
    n = max(len(decompose(exp.replace(" ", ""))), 1)
    nbad = sum(max(len(e.exp), len(e.act)) for e in errs)
    acc = max(0.0, 1.0 - nbad / n)
    return {"correct": acc >= 1.0, "accuracy": round(acc, 3),
            "expected": exp, "response": got, "errors": errs}


_KNUM = {"하나": "1", "둘": "2", "셋": "3", "넷": "4", "다섯": "5",
         "여섯": "6", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5"}


def _num(s):
    s = s.strip()
    if s.isdigit():
        return s
    return _KNUM.get(s, s)


if __name__ == "__main__":
    tasks = [
        make_task("syllable", "count", "자동차"),
        make_task("syllable", "blend", "", parts=["연", "필"]),
        make_task("syllable", "delete", "축구", target="구"),
        make_task("syllable", "add", "바지", syl="청", pos="front"),
        make_task("syllable", "reverse", "나무"),
        make_task("syllable", "substitute", "호박", old="호", new="수"),
        make_task("phoneme", "count", "강"),
        make_task("phoneme", "blend", "", jamos=["ㅂ", "ㅓ"]),
        make_task("phoneme", "delete", "기"),
        make_task("phoneme", "substitute", "저", new_onset="ㅊ"),
    ]
    for t in tasks:
        print(f"[{t.level:8}/{t.op:10}] {t.prompt:28} 기대={t.expected}")
    print()
    # 채점 시연: 도치 과제에 도치 실패(원형 그대로) 응답
    t = make_task("syllable", "reverse", "나무")
    print("도치 정답:", score_response(t, "무나")["correct"])
    print("도치 실패:", score_response(t, "나무")["correct"], "(가방→바강 류 약점 신호)")
