"""고정 문항(단어+문장)의 G2P 발음을 미리 계산해 pron_cache.json으로 저장.

HF Spaces 등 mecab(eunjeon) 없는 환경에서 g2pkk 없이도 동작하게 한다.
로컬(g2pkk 작동 환경)에서 1회 실행해 캐시를 만들고 함께 배포한다.
"""
import json
import os

from g2pkk import G2p

HERE = os.path.dirname(os.path.abspath(__file__))
g2p = G2p()

with open(os.path.join(HERE, "items.json"), encoding="utf-8") as f:
    data = json.load(f)

keys = set()
for it in data.get("items", []):
    keys.add(it["word"])
for s in data.get("sentences", []):
    keys.add(s["text"])

cache = {k: g2p(k) for k in sorted(keys)}
with open(os.path.join(HERE, "pron_cache.json"), "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print(f"cached {len(cache)} entries -> pron_cache.json")
for k, v in list(cache.items())[:8]:
    print(f"  {k} -> {v}")
