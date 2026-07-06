"""한글 발음 → slplab 음소(로마자) 변환.

slplab/wav2vec2-xls-r-300m_phone-mfa_korean 의 vocab(45 phones)에 맞춰
G2P 발음(한글)을 음소열로 바꾼다. 모델 출력과 표기가 같아 바로 비교된다.

모델 실측으로 맞춰본 예: 궁물→G U NG M U L, 낟→N A t, 멀튼→M EO L Th EU N,
바다→B A D A, 강→G A NG.
"""

from align import Jamo, decompose

# 초성
ONSET = {
    "ㄱ": "G", "ㄲ": "GG", "ㅋ": "Kh", "ㄴ": "N", "ㄷ": "D", "ㄸ": "DD",
    "ㅌ": "Th", "ㄹ": "R", "ㅁ": "M", "ㅂ": "B", "ㅃ": "BB", "ㅍ": "Ph",
    "ㅅ": "S", "ㅆ": "SS", "ㅇ": "", "ㅈ": "J", "ㅉ": "JJ", "ㅊ": "CHh", "ㅎ": "H",
}
# 중성(모음)
VOWEL = {
    "ㅏ": "A", "ㅐ": "E", "ㅑ": "iA", "ㅒ": "iE", "ㅓ": "EO", "ㅔ": "E",
    "ㅕ": "iEO", "ㅖ": "iE", "ㅗ": "O", "ㅘ": "oA", "ㅙ": "oE", "ㅚ": "oE",
    "ㅛ": "iO", "ㅜ": "U", "ㅝ": "uEO", "ㅞ": "oE", "ㅟ": "uI", "ㅠ": "iU",
    "ㅡ": "EU", "ㅢ": "euI", "ㅣ": "I",
}
# 종성(7종 중화). 겹받침은 대표음으로.
CODA = {
    "ㄱ": "k", "ㄲ": "k", "ㅋ": "k", "ㄳ": "k", "ㄺ": "k",
    "ㄴ": "N", "ㄵ": "N", "ㄶ": "N",
    "ㄷ": "t", "ㅅ": "t", "ㅆ": "t", "ㅈ": "t", "ㅊ": "t", "ㅌ": "t", "ㅎ": "t",
    "ㄹ": "L", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㅀ": "L",
    "ㅁ": "M", "ㄻ": "M",
    "ㅂ": "p", "ㅍ": "p", "ㅄ": "p", "ㄿ": "p",
    "ㅇ": "NG",
}


def to_phone_tokens(pron_text):
    """G2P 발음(한글) → 음소 Jamo 토큰 리스트(char=phone, role 보존).
    role은 transparent 종성 관대 채점에 사용된다(onset/nucleus/coda)."""
    out = []
    for j in decompose(pron_text):
        if j.role == "onset":
            ph = ONSET.get(j.char, j.char)
        elif j.role == "nucleus":
            ph = VOWEL.get(j.char, j.char)
        elif j.role == "coda":
            ph = CODA.get(j.char, j.char)
        else:
            ph = j.char
        if ph:  # 무음 초성 ㅇ 은 건너뜀
            out.append(Jamo(ph, j.role, j.syl))
    return out


def phones_from_model(text):
    """모델 출력 문자열(공백 구분 음소) → Jamo 토큰 리스트(role 미상='')."""
    return [Jamo(p, "", i) for i, p in enumerate(text.split())]


def phone_str(tokens):
    return " ".join(t.char for t in tokens)


# ---- 음소 → 한글 복원 (사람이 읽을 수 있게) ----
_P2ONSET = {
    "G": "ㄱ", "GG": "ㄲ", "Kh": "ㅋ", "N": "ㄴ", "D": "ㄷ", "DD": "ㄸ",
    "Th": "ㅌ", "R": "ㄹ", "L": "ㄹ", "M": "ㅁ", "B": "ㅂ", "BB": "ㅃ",
    "Ph": "ㅍ", "S": "ㅅ", "SS": "ㅆ", "J": "ㅈ", "JJ": "ㅉ", "CHh": "ㅊ", "H": "ㅎ",
}
_P2VOWEL = {v: k for k, v in VOWEL.items()}  # 'A'->'ㅏ' 등 (첫 매핑 우선)
_P2CODA = {"k": "ㄱ", "t": "ㄷ", "p": "ㅂ", "N": "ㄴ", "M": "ㅁ", "NG": "ㅇ", "L": "ㄹ"}
_CODA_ONLY = {"k", "t", "p", "NG"}
_VOWELS = set(VOWEL.values())

_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def _compose(cho, jung, jong=""):
    """초/중/종 자모 → 한글 음절."""
    try:
        ci, ji = _CHO.index(cho or "ㅇ"), _JUNG.index(jung)
        ki = _JONG.index(jong) if jong else 0
        return chr(0xAC00 + (ci * 21 + ji) * 28 + ki)
    except ValueError:
        return (cho or "") + (jung or "") + (jong or "")


def phone_label(p):
    """음소 기호 → 사람이 읽을 한글 자모(표시용). 예 'EU'→'ㅡ', 'S'→'ㅅ', 'k'→'ㄱ(받침)'."""
    if p in _P2VOWEL:
        return _P2VOWEL[p]
    if p in _P2ONSET:
        return _P2ONSET[p]
    if p in _P2CODA:
        return _P2CODA[p] + "(받침)"
    return p


def phones_to_hangul(phones):
    """음소열(list[str] 또는 공백문자열) → 한글 문자열(들린 대로 복원)."""
    if isinstance(phones, str):
        phones = phones.split()
    out, i, n = [], 0, len(phones)
    while i < n:
        p = phones[i]
        if p in _VOWELS:
            onset, nucleus = "", p
            i += 1
        elif (i + 1 < n) and (phones[i + 1] in _VOWELS) and p not in _CODA_ONLY:
            onset, nucleus = p, phones[i + 1]
            i += 2
        else:
            # 모음 없는 자음(겹침/누락) → 낱자모로 표기
            out.append(_P2ONSET.get(p, _P2CODA.get(p, p)))
            i += 1
            continue
        coda = ""
        if i < n and phones[i] not in _VOWELS:
            nxt = phones[i + 1] if i + 1 < n else None
            if phones[i] in _CODA_ONLY or nxt is None or nxt not in _VOWELS:
                if phones[i] in _P2CODA:
                    coda = phones[i]
                    i += 1
        out.append(_compose(_P2ONSET.get(onset, ""), _P2VOWEL.get(nucleus, nucleus),
                            _P2CODA.get(coda, "")))
    return "".join(out)


if __name__ == "__main__":
    from g2p_expected import expected_prons

    cases = [("낟", "nonword"), ("풉", "nonword"), ("멀튼", "nonword"),
             ("국물", "phonrule"), ("꽃을", "phonrule"), ("해돋이", "phonrule"),
             ("바다", "transparent"), ("구두", "transparent"), ("강", "transparent")]
    for w, t in cases:
        pron = expected_prons(w, t)[0]
        print(f"{w:6}({pron}) -> {phone_str(to_phone_tokens(pron))}")
