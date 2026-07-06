"""브라우저 마이크로 녹음·채점하는 웹앱 (Gradio).

이 데스크탑엔 마이크가 없어도 된다: 녹음은 '접속한 기기의 브라우저'에서 하고,
오디오만 서버로 업로드돼 ASR+채점이 돈다. share=True 로 임시 공개 링크 생성.

실행:
    python webapp.py
콘솔에 뜨는 https://....gradio.live 링크를 폰/노트북에서 열어 사용.
"""

import json
import os
import shutil

import gradio as gr
import librosa
import numpy as np

from asr import transcribe  # filepath -> 16k mono 강제 로드 후 CTC 디코딩
from g2p_expected import expected_prons
from scoring import score_word

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "recordings")
os.makedirs(REC_DIR, exist_ok=True)
_counter = {"n": 0}

with open(os.path.join(HERE, "items.json"), encoding="utf-8") as f:
    _data = json.load(f)
ITEMS = _data.get("items", [])
REAL_WORDS = set(_data.get("real_words", []))
SENTENCES = _data.get("sentences", [])

# 드롭다운 라벨 -> (word, type)
CHOICES = {f"{it['word']}  ({it['type']})": (it["word"], it["type"]) for it in ITEMS}
SENT_CHOICES = {f"[{s['level']}] {s['text']}": s["text"] for s in SENTENCES}

EXPLAINER_MD = r"""
# 이 시스템은 어떻게 만들어졌나

## 1. 핵심 아이디어 — "전사"가 아니라 "정답 대비 채점"
읽어야 할 단어·문장을 **이미 알고 있습니다.** 그래서 "무슨 말을 하려 했나"(자유 전사)가
아니라 **"이 정해진 음소열을 얼마나 정확히 읽었나"**를 측정합니다.

- **Whisper 안 씀**: Whisper류는 디코더가 강한 언어모델이라 아이가 [채상]이라 읽어도
  문맥상 '책상'으로 **자동 보정**해버립니다. 읽기평가에선 그 오독이 진단 신호인데
  숨겨버리므로 치명적. → 소리에 충실한 **CTC 모델(wav2vec2)** 만 사용.
- **학습 0**: 공개된 사전학습 모델을 **추론만** 합니다. 채점은 전부 규칙 코드.

## 2. 처리 흐름
```
[음성 16kHz] ─(① 음향 인식, CTC)→ 음소열
[정답 단어]  ─(② G2P)→ 기대 발음 ─→ 기대 음소열
                  │
       (③ 정렬: 단어/어절별 구간 배정)
                  │
       (④ 채점: 문항 유형별 규칙 + GOP 신뢰도)
                  │
        단어별 정·오 / 정확도 / (문장은) 읽기 속도
```

## 3. 각 단계
- **① 음향 인식(ASR)**: 두 모델 제공. *음소 모델*(slplab, 권장)은 한국어 음소를 직접
  인식 — 1음절·짧은 단어에 강하고 음소 단위로 정밀. *한글 모델*(kresnik)은 비교용.
  오디오는 **무조건 16kHz mono**로 변환(가장 흔한 실패 원인).
- **② G2P(정답→발음)**: `g2pkk`로 철자를 실제 발음으로. 예) 국물→[궁물](비음화),
  꽃을→[꼬츨](연음), 해돋이→[해도지](구개음화). 문장은 어절 넘는 연음까지 반영.
- **③ 정렬**: 실제 발화 음소열을 단어/어절 순서대로 **DP로 최적 분할**해 배정.
  1음절 단어가 누락돼 뒤가 밀려도 각 단어에 맞는 구간을 찾아 흡수.
- **④ 채점**: **문항 유형별로 엄격도를 다르게** 적용 —
  | 유형 | 측정 | 채점 |
  |---|---|---|
  | 무의미단어 | 순수 해독력 | 가장 엄격 (맥락 추측 불가) |
  | 음운규칙 단어 | 규칙 적용력 | 규칙 실패를 오류로 |
  | 자소-음소일치 | 기본 자모 대응 | 받침 단순 오류는 관대 |

## 4. GOP — "제대로 읽었는데 틀림"을 줄이는 장치
모델이 시→지처럼 비슷한 소리로 잘못 출력할 때가 있습니다. 그래서 **자유 인식 결과로
채점하지 않고**, 기대 음소를 음향에 **강제정렬해 "그 소리를 실제로 냈는지"의 신뢰도
(GOP)**를 직접 측정해 판정합니다.
- **GOP 높음** = 또렷이 발음함 → 정답 (모델 출력이 틀려도 구제)
- **GOP 낮음** = 그 소리를 못 들음 → **⚠확인필요**(오독일 수도, 인식 한계일 수도)

## 5. 검사 모드 ↔ 실제 임상
- **목록 읽기** = 실제 **해독 검사**(단어/무의미단어 목록) — 난독증 변별력 1순위
- **문장 읽기** = 실제 **읽기 유창성 검사**(정확도 + 속도)
- 단어 1개 = 참고용

## 6. 솔직한 한계 (이 도구는 '채점 엔진'이지 '진단기'가 아님)
난독증 '진단'에는 추가로 필요합니다:
- **또래 규준 비교**(백분위) — 정상 아동 데이터 필요 (현재는 원점수만)
- **음운인식·작업기억·빠른이름대기** 등 음운처리 검사 (별도 과제)
- **임계값 캘리브레이션** — GOP 임계값 등을 임상가 채점본과 대조해 보정
- **파인튜닝** — 정확도 도약은 결국 목표 화자(아동) 음성으로 모델 적응

> 즉 지금 도구 = 평가자의 '귀 + 채점지'를 자동화한 **해독/유창성 정확도 채점 엔진**입니다.
"""


def quality_warn(q):
    """품질 게이트: 문제 있으면 안내문, 없으면 None."""
    if q["peak"] < 0.05:
        return f"⚠ 녹음이 너무 작습니다 (peak={q['peak']:.3f}). 마이크에 더 가까이, 또렷하게 다시 읽어주세요."
    if q.get("clip", 0) > 0.005 or q["peak"] >= 0.99:
        return (f"⚠ 녹음이 너무 큽니다(찌그러짐, peak={q['peak']:.3f}). "
                f"마이크에서 조금 떨어지거나 볼륨을 낮춰 다시 읽어주세요.")
    if q["voiced_sec"] < 0.3:
        return f"⚠ 발화가 너무 짧습니다 ({q['voiced_sec']:.2f}s). 끝까지 또렷하게 읽어주세요."
    return None


def evaluate(choice, custom_word, custom_type, audio_path, model_choice="음소모델(권장)"):
    if audio_path is None:
        return "⚠ 먼저 마이크로 녹음하세요.", ""

    if custom_word and custom_word.strip():
        word, item_type = custom_word.strip(), custom_type
    elif choice in CHOICES:
        word, item_type = CHOICES[choice]
    else:
        return "⚠ 단어를 선택하거나 직접 입력하세요.", ""

    _counter["n"] += 1
    saved = os.path.join(REC_DIR, f"{_counter['n']:03d}_{word}.wav")
    try:
        shutil.copy(audio_path, saved)
    except Exception:
        saved = audio_path
    from asr import audio_quality, load_audio
    q = audio_quality(load_audio(audio_path))
    diag = (f"[오디오] 발화 {q['voiced_sec']:.2f}s, peak={q['peak']:.3f}, "
            f"저장={os.path.basename(saved)}")
    warn = quality_warn(q)
    if warn:
        return f"{warn}\n{diag}", ""

    item = {"word": word, "type": item_type}
    if model_choice.startswith("음소"):
        from phoneme_asr import transcribe_phones
        from pipeline_phoneme import score_list_phoneme
        actual = transcribe_phones(audio_path)
        scores, heard = score_list_phoneme([item], actual)
        s, h = scores[0], heard[0]
    else:
        from asr import transcribe
        from pipeline_list import score_list
        from align import align_words_dp, tokens_to_text
        actual = transcribe(audio_path).strip()
        scores, prons = score_list([item], actual)
        s = scores[0]
        segs = align_words_dp(prons, actual)
        h = tokens_to_text(segs[0])

    mark = "✅ 정답" if s.correct else "❌ 오류"
    head = (
        f"[{model_choice}]\n"
        f"읽을 단어: 「{word}」  (유형 {item_type})\n"
        f"들린대로: 「{h or '(못 들음)'}」\n"
        f"채점: {mark}   정확도 {s.accuracy:.0%}\n"
        f"{diag}"
    )
    lines = []
    for e in s.errors:
        ej = "".join(t.char for t in e.exp) or "-"
        aj = "".join(t.char for t in e.act) or "-"
        lines.append(f"- {e.kind} [{e.tag}] {ej} → {aj}  (감점 {e.weight})")
    detail = "\n".join(lines) if lines else "(차이 없음)"
    return head, detail


# ---- 트랙1: 목록(연결발화) 읽기 ----
# 전체 32개는 한 번에 읽기 무리 → 유형별 3개씩 대표 9개만 목록 모드에 사용.
def _pick(t, k):
    return [it for it in ITEMS if it["type"] == t][:k]


LIST_ITEMS = _pick("nonword", 3) + _pick("phonrule", 3) + _pick("transparent", 3)
LIST_PROMPT = "   ".join(it["word"] for it in LIST_ITEMS)


def evaluate_list(audio_path, model_choice):
    """전체 단어 목록을 이어 읽은 1개 녹음 → 단어별 채점.
    model_choice: '음소모델(권장)' | '한글모델(kresnik)'."""
    if audio_path is None:
        return "⚠ 먼저 목록 전체를 이어 읽어 녹음하세요.", ""

    from asr import audio_quality, load_audio

    _counter["n"] += 1
    saved = os.path.join(REC_DIR, f"list_{_counter['n']:03d}.wav")
    try:
        shutil.copy(audio_path, saved)
    except Exception:
        saved = audio_path
    q = audio_quality(load_audio(audio_path))
    warn = quality_warn(q)
    if warn:
        return f"{warn}\n발화 {q['voiced_sec']:.1f}s, peak={q['peak']:.3f}", ""

    use_phoneme = model_choice.startswith("음소")
    if use_phoneme:
        # 주채점 = GOP(강제정렬 신뢰도). 들린대로는 자유인식으로 참고 표시.
        from phoneme_asr import transcribe_phones
        from phoneme_map import phone_label, phones_to_hangul
        from pipeline_phoneme import score_list_phoneme
        from gop import score_list_gop
        actual = transcribe_phones(audio_path)
        _, heard = score_list_phoneme(LIST_ITEMS, actual)
        heard_all = phones_to_hangul(actual)
        gres, _ = score_list_gop(LIST_ITEMS, audio_path)

        n_ok = sum(1 for r in gres if r["verdict"] == "correct")
        n_unc = sum(1 for r in gres if r["verdict"] != "correct")
        head = (
            f"[{model_choice} · GOP 주채점]\n"
            f"들린 전체(참고): 「{heard_all}」\n"
            f"읽을 단어: 「{LIST_PROMPT}」\n"
            f"발화 {q['voiced_sec']:.1f}s, peak={q['peak']:.3f}\n"
            f"=== 정답 {n_ok}/{len(gres)}"
            + (f" · 확인필요 {n_unc}" if n_unc else "") + " ==="
        )
        mk = {"correct": "✅", "uncertain": "⚠확인필요", "review": "⚠검수(오류가능)"}
        lines = []
        for r, h in zip(gres, heard):
            weak = ", ".join(f"{phone_label(p)}({g:.1f})" for p, g, _ in r["weak"]) or "-"
            lines.append(
                f"{mk[r['verdict']]} {r['word']} ({r['type']})  정확도 {r['accuracy']:.0%}"
                f"  들린:{h or '-'}  약한소리:{weak}"
            )
        note = ("\n\n※ 주채점은 GOP(기대 음소를 또렷이 냈는지)다. ⚠ = 신뢰 낮음(오독일 수도,"
                " 인식 한계일 수도) → 검수. 임계값은 잠정값(임상가 채점본으로 보정 필요).")
        return head, "\n".join(lines) + (note if n_unc else "")

    # 한글 모델: 기존 free-decode 방식
    from asr import transcribe
    from pipeline_list import score_list
    from align import align_words_dp, tokens_to_text
    actual = transcribe(audio_path).strip()
    scores, prons = score_list(LIST_ITEMS, actual)
    segs = align_words_dp(prons, actual)
    heard = [tokens_to_text(seg) for seg in segs]
    n_ok = sum(1 for s in scores if s.correct)
    head = (
        f"[{model_choice}]\n"
        f"들린 전체: 「{actual}」\n"
        f"읽을 단어: 「{LIST_PROMPT}」\n"
        f"발화 {q['voiced_sec']:.1f}s, peak={q['peak']:.3f}\n"
        f"=== {n_ok}/{len(scores)} 정답 ==="
    )
    lines = [f"{'✅' if s.correct else '❌'} {s.word} ({s.item_type})  →  들린대로: {h or '(못 들음)'}"
             f"   정확도 {s.accuracy:.0%}" for s, h in zip(scores, heard)]
    return head, "\n".join(lines)


def evaluate_sentence(audio_path, sent_choice):
    """문장/지문을 읽은 녹음 → 어절별 정확도 채점(음소 모델)."""
    if audio_path is None:
        return "⚠ 먼저 문장을 소리 내어 읽어 녹음하세요.", ""
    sentence = SENT_CHOICES.get(sent_choice)
    if not sentence:
        return "⚠ 문장을 선택하세요.", ""

    from asr import audio_quality, load_audio
    from phoneme_asr import transcribe_phones
    from phoneme_map import phones_to_hangul
    from pipeline_sentence import format_sentence_report, score_sentence

    _counter["n"] += 1
    saved = os.path.join(REC_DIR, f"sent_{_counter['n']:03d}.wav")
    try:
        shutil.copy(audio_path, saved)
    except Exception:
        saved = audio_path
    q = audio_quality(load_audio(audio_path))
    warn = quality_warn(q)
    if warn:
        return f"{warn}\n발화 {q['voiced_sec']:.1f}s, peak={q['peak']:.3f}", ""

    actual = transcribe_phones(audio_path)
    scores, heard, _ = score_sentence(sentence, actual)

    # 주채점 = GOP(어절별). 어절을 transparent로 보고 받침 관대.
    from gop import score_list_gop
    from phoneme_map import phone_label
    from pipeline_sentence import surface_eojeols
    eojeols = surface_eojeols(sentence)
    gres, _ = score_list_gop([{"word": e, "type": "transparent"} for e in eojeols],
                             audio_path)

    n_ok = sum(1 for r in gres if r["verdict"] == "correct")
    n_unc = sum(1 for r in gres if r["verdict"] != "correct")

    from fluency import fluency_metrics, format_fluency
    fm = fluency_metrics(sentence, scores, q["voiced_sec"])
    head = (
        f"문장: 「{sentence}」\n"
        f"들린 전체(참고): 「{phones_to_hangul(actual)}」\n"
        f"=== 어절 정답 {n_ok}/{len(gres)} ({n_ok/max(len(gres),1):.0%})"
        + (f" · 확인필요 {n_unc}" if n_unc else "") + " ===\n"
        f"[유창성] {format_fluency(fm)}\n"
        f"(peak={q['peak']:.3f}, 저장={os.path.basename(saved)})"
    )
    mk = {"correct": "✅", "uncertain": "⚠확인필요", "review": "⚠검수(오류가능)"}
    lines = []
    for r, h in zip(gres, heard):
        weak = ", ".join(f"{phone_label(p)}({g:.1f})" for p, g, _ in r["weak"]) or "-"
        lines.append(f"{mk[r['verdict']]} {r['word']}  정확도 {r['accuracy']:.0%}"
                     f"  들린:{h or '-'}  약한소리:{weak}")
    return head, "\n".join(lines)


with gr.Blocks(title="난독증 읽기평가 v0") as demo:
    gr.Markdown("## 한국어 난독증 읽기평가 (v0)")
    with gr.Tabs():
        with gr.Tab("목록 읽기 (권장)"):
            gr.Markdown(
                "아래 단어들을 **한 호흡에 또박또박 이어서** 읽고 '채점하기'를 누르세요.\n"
                "연결발화라 모델이 짧은단어 사각지대를 피해 잘 인식합니다.\n\n"
                f"### 읽을 목록\n# {LIST_PROMPT}"
            )
            lmodel = gr.Radio(
                ["음소모델(권장)", "한글모델(kresnik)"],
                label="인식 모델", value="음소모델(권장)",
            )
            laudio = gr.Audio(sources=["microphone"], type="filepath", label="목록 전체 녹음")
            lbtn = gr.Button("채점하기", variant="primary")
            lout_head = gr.Textbox(label="결과", lines=6)
            lout_detail = gr.Textbox(label="단어별 채점", lines=11)
            lbtn.click(evaluate_list, [laudio, lmodel], [lout_head, lout_detail])

        with gr.Tab("문장 읽기"):
            gr.Markdown(
                "문장/지문을 **자연스럽게 소리 내어** 읽고 채점하세요. "
                "어절(띄어쓰기 단위)별 정확도가 나옵니다. 문장 전체 연음까지 반영합니다."
            )
            sent_choice = gr.Dropdown(
                list(SENT_CHOICES.keys()), label="문장 선택",
                value=list(SENT_CHOICES.keys())[0] if SENT_CHOICES else None,
            )
            saudio2 = gr.Audio(sources=["microphone"], type="filepath", label="문장 녹음")
            sbtn2 = gr.Button("채점하기", variant="primary")
            sout_head = gr.Textbox(label="결과", lines=5)
            sout_detail = gr.Textbox(label="어절별 채점", lines=9)
            sbtn2.click(evaluate_sentence, [saudio2, sent_choice], [sout_head, sout_detail])

        with gr.Tab("단어 1개"):
            gr.Markdown(
                "단어 하나만 읽기. **음소 모델은 단어 1개에도 비교적 잘 인식**합니다. "
                "너무 크게(찌그러짐)·너무 작게 녹음하지 마세요."
            )
            with gr.Row():
                with gr.Column():
                    smodel = gr.Radio(
                        ["음소모델(권장)", "한글모델(kresnik)"],
                        label="인식 모델", value="음소모델(권장)",
                    )
                    choice = gr.Dropdown(
                        list(CHOICES.keys()), label="문항 선택",
                        value=list(CHOICES.keys())[0]
                    )
                    gr.Markdown("— 또는 직접 입력 —")
                    custom_word = gr.Textbox(label="직접 단어 입력(선택)", placeholder="예: 책상")
                    custom_type = gr.Radio(
                        ["nonword", "phonrule", "transparent"],
                        label="유형(직접 입력 시)", value="transparent",
                    )
                    audio = gr.Audio(sources=["microphone"], type="filepath", label="녹음")
                    btn = gr.Button("채점하기", variant="primary")
                with gr.Column():
                    out_head = gr.Textbox(label="결과", lines=6)
                    out_detail = gr.Textbox(label="오류 상세", lines=8)
            btn.click(evaluate, [choice, custom_word, custom_type, audio, smodel],
                      [out_head, out_detail])

        with gr.Tab("ℹ️ 원리 설명"):
            gr.Markdown(EXPLAINER_MD)


if __name__ == "__main__":
    import os as _os
    # 기본은 로컬 7860에 바인딩(외부 노출은 cloudflared 터널이 담당).
    # GRADIO_SHARE=1 이면 gradio.live 공유도 사용.
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(_os.environ.get("GRADIO_PORT", "7860")),
        share=_os.environ.get("GRADIO_SHARE", "0") == "1",
    )
