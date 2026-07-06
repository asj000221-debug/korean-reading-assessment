"""정답 철자 → 기대 발음(G2P).

비교는 반드시 발음 기준으로 해야 한다. 철자로 비교하면 정상 음운변동을 죄다
오류로 잡는다.

문항 유형별 허용 발음 후보만 여기서 만들고, 최종 엄격도는 scoring.py가 정한다.
- transparent: 규칙 적용형이 기준이되, 미적용 표면형(철자)도 후보로 허용
  (받침 조음 미숙 관대 처리는 scoring에서)
- phonrule: 규칙 적용 발음만 정답. 미적용을 정답으로 두면 측정 대상이 사라진다
- nonword: 맥락 추측 불가 → 철자 그대로 읽는 게 정답
"""

import json
import os
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))

# 사전 계산 발음 캐시(고정 문항). mecab(eunjeon) 없는 환경에서 g2pkk 없이 동작.
try:
    with open(os.path.join(_HERE, "pron_cache.json"), encoding="utf-8") as _f:
        _PRON_CACHE = json.load(_f)
except FileNotFoundError:
    _PRON_CACHE = {}


@lru_cache(maxsize=1)
def _g2p():
    """g2pkk는 mecab(eunjeon)에 의존 → 지연 로드. 없으면 None."""
    try:
        from g2pkk import G2p
        return G2p()
    except Exception:
        return None


def g2p(word):
    """철자 → 음운규칙 적용 발음(한글).

    1) 캐시 우선(고정 문항). 2) g2pkk 시도. 3) 둘 다 실패 시 철자 그대로(degrade).
    """
    if word in _PRON_CACHE:
        return _PRON_CACHE[word]
    engine = _g2p()
    if engine is not None:
        try:
            return engine(word)
        except Exception:
            pass
    return word  # G2P 불가 환경의 폴백(직접 입력 단어 등)


def expected_prons(word, item_type):
    """문항 유형별 '허용 정답 발음 후보' 리스트를 반환한다(중복 제거, 순서 유지).

    반환 리스트의 0번째가 '기준 발음(canonical)'이며, scoring은 보통 이것과 정렬한다.
    transparent는 표면형(철자)도 허용 후보로 추가한다.
    """
    rule_pron = g2p(word)

    if item_type == "phonrule":
        cands = [rule_pron]
    elif item_type == "transparent":
        # 규칙 적용형이 기준, 미적용 표면형(철자)도 허용
        cands = [rule_pron, word]
    elif item_type == "nonword":
        # 무의미단어: 철자 그대로 읽는 것이 정답. g2p가 바꿔도 표면형을 기준으로.
        cands = [word, rule_pron]
    else:
        raise ValueError(f"unknown item_type: {item_type!r}")

    # 중복 제거(순서 유지)
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


if __name__ == "__main__":
    samples = [
        ("국물", "phonrule"),
        ("꽃을", "phonrule"),
        ("바다", "transparent"),
        ("구두", "transparent"),
        ("낟", "nonword"),
        ("풉", "nonword"),
        ("멀튼", "nonword"),
    ]
    for w, t in samples:
        print(f"{w:6} [{t:11}] -> {expected_prons(w, t)}")
