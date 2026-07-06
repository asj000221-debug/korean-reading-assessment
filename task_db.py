# -*- coding: utf-8 -*-
"""과제 정답 DB (공개용 목업 문항).

표준화 검사 매뉴얼의 구조(3과제·타입 규칙·산출 구분)는 따르되, 문항 자체는 검사
보안 때문에 공개용 목업으로 갈아끼웠다. 정답의 single source of truth이고,
ASR이 필요한 세 과제(낱말·단락·음운인식-합성)만 담는다(나머지 하위검사는 범위 밖).

낱말 item은 (word 철자, type 엄격도, pron 채점기준 발음). pron은 g2p 결과를 박아둔다
(mecab 없는 배포환경에서도 동작하게 사전 계산). 무의미낱말도 사전이 없어 규칙형을 기준.

type 규칙(차등 엄격도):
  - 의미낱말 g2p(word)==word → transparent (받침 조음미숙 관대)
  - 의미낱말 g2p(word)!=word → phonrule (음운규칙 실패도 오류; 엄격)
  - 무의미낱말 전부 → nonword (맥락추측 불가, 가장 엄격)
"""

# ── ① 낱말읽기: 의미 40 + 무의미 40 (목업) ──
# (word, type, pron)  — 무의미는 의미와 1:1 최소대립쌍(채점 캘리브레이션 골든셋).
_MEANING = [
    ("눈", "transparent", "눈"),
    ("밤", "transparent", "밤"),
    ("길", "transparent", "길"),
    ("산", "transparent", "산"),
    ("별", "transparent", "별"),
    ("꽃", "phonrule", "꼳"),
    ("잎", "phonrule", "입"),
    ("문", "transparent", "문"),
    ("숲", "phonrule", "숩"),
    ("배", "transparent", "배"),
    ("나무", "transparent", "나무"),
    ("바다", "transparent", "바다"),
    ("구름", "transparent", "구름"),
    ("노래", "transparent", "노래"),
    ("국물", "phonrule", "궁물"),
    ("학교", "phonrule", "학꾜"),
    ("같이", "phonrule", "가치"),
    ("맏이", "phonrule", "마지"),
    ("낚시", "phonrule", "낙씨"),
    ("국민", "phonrule", "궁민"),
    ("코끼리", "transparent", "코끼리"),
    ("피아노", "transparent", "피아노"),
    ("무지개", "transparent", "무지개"),
    ("자전거", "transparent", "자전거"),
    ("해돋이", "phonrule", "해도지"),
    ("미역국", "phonrule", "미역꾹"),
    ("병아리", "transparent", "병아리"),
    ("도깨비", "transparent", "도깨비"),
    ("텔레비전", "transparent", "텔레비전"),
    ("해바라기", "transparent", "해바라기"),
    ("미끄럼틀", "transparent", "미끄럼틀"),
    ("장난감", "transparent", "장난감"),
    ("냉장고", "transparent", "냉장고"),
    ("숟가락", "phonrule", "숟까락"),
    ("젓가락", "phonrule", "젇까락"),
    ("꽃다발", "phonrule", "꼳따발"),
    ("아이스크림", "transparent", "아이스크림"),
    ("고슴도치", "transparent", "고슴도치"),
    ("미용실", "transparent", "미용실"),
    ("옥수수", "phonrule", "옥쑤수"),
]

_NONSENSE = [
    ("둔", "nonword", "둔"),
    ("밥", "nonword", "밥"),
    ("낄", "nonword", "낄"),
    ("삽", "nonword", "삽"),
    ("뻘", "nonword", "뻘"),
    ("꼿", "nonword", "꼳"),
    ("옆", "nonword", "엽"),
    ("뭄", "nonword", "뭄"),
    ("쑵", "nonword", "쑵"),
    ("빼", "nonword", "빼"),
    ("다무", "nonword", "다무"),
    ("바더", "nonword", "바더"),
    ("구릅", "nonword", "구릅"),
    ("도래", "nonword", "도래"),
    ("굽물", "nonword", "굼물"),
    ("합교", "nonword", "합꾜"),
    ("가티", "nonword", "가티"),
    ("먇이", "nonword", "먀지"),
    ("낚씨", "nonword", "낙씨"),
    ("국민", "nonword", "궁민"),
    ("토끼리", "nonword", "토끼리"),
    ("비아노", "nonword", "비아노"),
    ("무디개", "nonword", "무디개"),
    ("자던거", "nonword", "자던거"),
    ("해둗이", "nonword", "해두지"),
    ("미역굽", "nonword", "미역꿉"),
    ("뱡아리", "nonword", "뱡아리"),
    ("도깨미", "nonword", "도깨미"),
    ("텔레비젼", "nonword", "텔레비전"),
    ("해마라기", "nonword", "해마라기"),
    ("미끄넘틀", "nonword", "미끄넘틀"),
    ("잔난감", "nonword", "잔난감"),
    ("낸장고", "nonword", "낸장고"),
    ("숟마락", "nonword", "순마락"),
    ("젓카락", "nonword", "젇카락"),
    ("꼿다발", "nonword", "꼳따발"),
    ("아이스크딤", "nonword", "아이스크딤"),
    ("고습도치", "nonword", "고습또치"),
    ("미용딜", "nonword", "미용딜"),
    ("옥쑤수", "nonword", "옥쑤수"),
]

# transparent '다른 실단어 됨' 판정용 실단어 집합(의미낱말 40개).
REAL_WORDS = {w for (w, _t, _p) in _MEANING}


def word_items(section="all"):
    """낱말읽기 문항 리스트. section: 'meaning'|'nonsense'|'all'.

    반환 원소: {"no", "word", "type", "pron", "section"}.
    """
    if section == "meaning":
        rows = [(i + 1, r, "의미") for i, r in enumerate(_MEANING)]
    elif section == "nonsense":
        rows = [(i + 41, r, "무의미") for i, r in enumerate(_NONSENSE)]
    else:
        rows = [(i + 1, r, "의미") for i, r in enumerate(_MEANING)]
        rows += [(i + 41, r, "무의미") for i, r in enumerate(_NONSENSE)]
    return [
        {"no": no, "word": w, "type": t, "pron": p, "section": sec}
        for (no, (w, t, p), sec) in rows
    ]


# ── ② 단락읽기: P1 (목업 지문) ──
_P1 = (
    "마을 도서관 앞마당이 아침부터 시끌시끌하였습니다. 다훈이는 창문을 열고 밖을 내다보았습니다. 아이들이 "
    "벌써 줄을 서 있었습니다. \"오늘은 이야기 잔치가 열리는 날입니다. 좋아하는 책을 한 권씩 가져오세요.\""
    " 초록 조끼를 입은 사서 선생님께서 큰 소리로 말씀하셨습니다. \"이야기 잔치는 책 속의 주인공이 되어 "
    "친구들에게 이야기를 들려주는 자리입니다. 가장 기억에 남는 책을 가지고 오세요. 장소는 도서관 "
    "뒷마당입니다.\" 다훈이는 잠시 생각에 잠겼습니다. '어떤 책이 제일 재미있었더라? 맞아! 바다 밑 여행 "
    "이야기가 있었지. 잠수함을 타고 깊은 바다를 여행하는 이야기였어. 이번 잔치의 주인공은 바로 나야.' "
    "다훈이는 이모께서 생일 선물로 주신 바다 밑 여행 책을 찾기 시작하였습니다. 밖에서는 아이들이 사서 "
    "선생님께 이것저것 여쭈어 보고 있었습니다. \"잔치는 언제 시작하나요?\" 서연이가 여쭈어 보았습니다. "
    "\"이야기 잔치는 토요일 오후에 시작합니다. 좋아하는 책을 꼭 가져오세요. 재미있는 이야기를 함께 나누어 "
    "봅시다. 가장 멋진 이야기를 들려준 사람에게는 책 선물을 드립니다.\" 사서 선생님께서는 이렇게 말씀하시며"
    " 도서관 안으로 들어가셨습니다."
)


def paragraph(pid="P1"):
    """단락 정답. 반환: {"id", "text", "n_eojeol"}. n_eojeol은 공백 어절 수(표시용)."""
    import re
    clean = re.sub(r"[.,!?…·\"'“”‘’()]", "", _P1)
    n = len([w for w in clean.split() if w])
    return {"id": pid, "text": _P1, "n_eojeol": n}


# ── ③ 음운인식(합성): 16 타깃, 닫힌집합 (목업) ──
# 전부 무의미 음절 → 아동이 어휘로 추측 못 함. 후보 제약 디코딩의 후보셋.
# bin: '2-3'(2·3음소) | '4-5'(4·5음소) — 매뉴얼 산출 구분.
_PA_SYNTH = [
    ("뚜", 2, "2-3"),
    ("갸", 2, "2-3"),
    ("쬬", 2, "2-3"),
    ("퓨", 2, "2-3"),
    ("캄", 3, "2-3"),
    ("톤", 3, "2-3"),
    ("넙", 3, "2-3"),
    ("횐", 3, "2-3"),
    ("다꼬", 4, "4-5"),
    ("프네", 4, "4-5"),
    ("코무", 4, "4-5"),
    ("뱌노", 4, "4-5"),
    ("토캉", 5, "4-5"),
    ("께분", 5, "4-5"),
    ("하뿔", 5, "4-5"),
    ("니껍", 5, "4-5"),
]


def pa_items():
    """음운인식(합성) 타깃 리스트(닫힌집합 후보 = 전체 목록).

    반환 원소: {"no", "target", "n_phone", "bin"}. target 발음 = 표기 그대로(무의미).
    """
    return [
        {"no": i + 1, "target": w, "n_phone": n, "bin": b}
        for i, (w, n, b) in enumerate(_PA_SYNTH)
    ]


def pa_candidates():
    """닫힌집합 후보 단어 리스트(문자열 16개)."""
    return [w for (w, _n, _b) in _PA_SYNTH]


# ── pron 캐시 병합용(다른 모듈이 g2p 없이도 동일 발음을 쓰게) ──
def all_prons():
    """word→pron 딕셔너리(낱말 80 + 음운인식 16). pron_cache.json 병합에 사용."""
    d = {}
    for w, _t, p in _MEANING + _NONSENSE:
        d[w] = p
    for w, _n, _b in _PA_SYNTH:
        d[w] = w
    return d


if __name__ == "__main__":
    mw = word_items("meaning")
    nw = word_items("nonsense")
    print(f"① 낱말읽기: 의미 {len(mw)} + 무의미 {len(nw)} = {len(mw)+len(nw)}문항")
    tally = {}
    for it in mw + nw:
        tally[it["type"]] = tally.get(it["type"], 0) + 1
    print("   타입 분포:", tally)
    p = paragraph()
    print(f"② 단락읽기: {p['id']}  어절 {p['n_eojeol']}개")
    pa = pa_items()
    b23 = sum(1 for x in pa if x["bin"] == "2-3")
    print(f"③ 음운인식(합성): {len(pa)}타깃 (2-3음소 {b23} / 4-5음소 {len(pa)-b23}), 후보={pa_candidates()}")
