# -*- coding: utf-8 -*-
"""3과제 Gradio 웹앱 — 브라우저 마이크로 녹음·채점.

터미널 오디오 세션 없이 브라우저에서 바로 돌려보는 프런트엔드. 녹음은 브라우저가,
채점은 서버가 slplab 음소-CTC로 한다. 채점 로직은 run_tasks.py / pa_synth.py 재사용
(SSOT=task_db.py). 탭은 낱말읽기 / 단락읽기 / 음운인식(합성) 세 개.

python app_tasks.py 로 로컬(7860), --share 붙이면 공유 링크. 첫 실행 시 slplab
음소모델(~1.2GB)을 HF Hub에서 자동 다운로드한다(수 분).
"""

# --- gradio_client 알려진 버그 패치(app.py와 동일) ---
import gradio_client.utils as _gcu

_orig_jstpt = _gcu._json_schema_to_python_type


def _safe_jstpt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_jstpt(schema, defs)


_gcu._json_schema_to_python_type = _safe_jstpt

import gradio as gr

import task_db


# ── 낱말 목록 / 지문 미리보기(읽을 내용 제시) ──
def words_prompt(section):
    items = task_db.word_items(section)
    words = [it["word"] for it in items]
    rows = ["".join(f"| {w} " for w in words[i:i + 10]) + "|"
            for i in range(0, len(words), 10)]
    sep = "|" + " --- |" * 10
    head = "| " + " | ".join(str(i + 1) for i in range(10)) + " |"
    return f"**아래 {len(words)}개 낱말을 위→아래 순서로 또박또박 읽으세요.**\n\n" \
           f"{head}\n{sep}\n" + "\n".join(rows)


def _err_md(e):
    """예외를 화면에 보이게(공백 대신). 긴 녹음 처리 실패 등을 사용자에게 표면화."""
    import traceback
    tb = traceback.format_exc(limit=3)
    return (f"### ⚠ 채점 중 오류\n`{type(e).__name__}: {e}`\n\n"
            f"<details><summary>자세히</summary>\n\n```\n{tb}\n```\n</details>")


# ══════════════ ① 낱말읽기 ══════════════
def grade_words(wav_path, section):
    if not wav_path:
        return "🎤 먼저 녹음(또는 업로드)하세요.", [], ""
    try:
        from phoneme_asr import transcribe_phones
        from phoneme_map import phones_to_hangul
        from run_tasks import score_words, summarize_words

        actual = transcribe_phones(wav_path)
        items, scores, heard = score_words(actual, section)
        rows = []
        for it, s, h in zip(items, scores, heard):
            rows.append([it["no"], it["section"], s.item_type, it["word"],
                         it["pron"], h or "(못들음)", f"{s.accuracy:.0%}",
                         "✅" if s.correct else "❌"])
        agg = summarize_words(items, scores)
        summ = "### 정답률\n" + "\n".join(
            f"- **{k}**: {v['correct']}/{v['total']} ({v['rate']:.0%})"
            for k, v in agg.items())
        heard_all = f"들린 전체: 「{phones_to_hangul(actual)}」"
        return heard_all, rows, summ
    except Exception as e:
        return _err_md(e), [], ""


# ══════════════ ② 단락읽기 ══════════════
def grade_para(wav_path):
    if not wav_path:
        return "🎤 먼저 녹음(또는 업로드)하세요.", [], ""
    try:
        from phoneme_asr import transcribe_phones
        from run_tasks import score_paragraph

        actual = transcribe_phones(wav_path)
        surf, scores, heard = score_paragraph(actual)
        rows = [[i + 1, s.word, h or "(못들음)", "✅" if s.correct else "❌"]
                for i, (s, h) in enumerate(zip(scores, heard))]
        n_ok = sum(1 for s in scores if s.correct)
        tot = len(scores)
        summ = f"### 어절 정확도: {n_ok}/{tot} ({(n_ok/tot if tot else 0):.0%})"
        return f"들린 어절 {tot}개 채점 완료", rows, summ
    except Exception as e:
        return _err_md(e), [], ""


# ══════════════ ③ 음운인식(합성) ══════════════
def grade_pa(wav_path, target):
    if not wav_path:
        return "🎤 먼저 녹음(또는 업로드)하세요.", []
    try:
        return _grade_pa_inner(wav_path, target)
    except Exception as e:
        return _err_md(e), []


def _grade_pa_inner(wav_path, target):
    import pa_synth

    v = pa_synth.verify(wav_path, target=target)
    mark = "✅ 정답" if v["correct"] else "❌ 오답"
    reject = {"wrong_candidate": "다른 후보로 합성됨(합성 실패)",
              "mispronounced": "목표는 맞췄으나 음소 오발음",
              None: "-"}.get(v.get("reject"), v.get("reject"))
    md = (f"## {mark}\n"
          f"- 목표: **{target}**  →  인식(pred): **{v['pred']}**\n"
          f"- 판정 사유: {reject}\n"
          f"- 목표 최저 GOP: `{v.get('min_gop')}`  (임계 {pa_synth.GOP_OK} 미만이면 오발음)\n")
    ranking = [[i + 1, w, f"{s:.4f}"] for i, (w, s) in enumerate(v["ranking"])]
    return md, ranking


# ══════════════ UI ══════════════
def build():
    cands = task_db.pa_candidates()
    with gr.Blocks(title="읽기검사 음성인식 3과제") as demo:
        gr.Markdown("# 읽기검사 음성인식·채점 — 3과제 러너\n"
                    "brower 마이크로 녹음 → slplab 음소-CTC 채점 (파인튜닝 없이 baseline)")

        # ① 낱말읽기
        with gr.Tab("① 낱말읽기"):
            sec = gr.Radio(["all", "meaning", "nonsense"], value="nonsense",
                           label="구간 (meaning=의미40 / nonsense=무의미40 / all=80)")
            prompt = gr.Markdown(words_prompt("nonsense"))
            sec.change(words_prompt, sec, prompt)
            w_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="목록을 다 읽고 정지")
            w_btn = gr.Button("채점", variant="primary")
            w_heard = gr.Markdown()
            w_summ = gr.Markdown()
            w_tbl = gr.Dataframe(
                headers=["no", "구간", "유형", "낱말", "기대발음", "들림", "정확도", "정/오"],
                label="낱말별 결과", wrap=True)
            w_btn.click(grade_words, [w_audio, sec], [w_heard, w_tbl, w_summ])

        # ② 단락읽기
        with gr.Tab("② 단락읽기"):
            p = task_db.paragraph()
            gr.Markdown(f"**아래 지문을 소리내어 읽으세요 (약 {p['n_eojeol']}어절).**\n\n{p['text']}")
            p_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="지문을 다 읽고 정지")
            p_btn = gr.Button("채점", variant="primary")
            p_heard = gr.Markdown()
            p_summ = gr.Markdown()
            p_tbl = gr.Dataframe(headers=["어절#", "표기", "들림", "정/오"],
                                 label="어절별 결과", wrap=True)
            p_btn.click(grade_para, p_audio, [p_heard, p_tbl, p_summ])

        # ③ 음운인식(합성)
        with gr.Tab("③ 음운인식(합성)"):
            gr.Markdown("검사자가 음소를 끊어 들려주면, 아동이 **합쳐서 한 번에** 발음하는 과제.\n"
                        f"타깃을 고르고 그 소리를 발음하세요. 후보 16개: {cands}")
            tgt = gr.Dropdown(cands, value=cands[0], label="목표 단어")
            a_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="목표를 발음하고 정지")
            a_btn = gr.Button("채점", variant="primary")
            a_md = gr.Markdown()
            a_tbl = gr.Dataframe(headers=["순위", "후보", "우도(프레임당 logprob)"],
                                 label="닫힌집합 후보 순위(top5)")
            a_btn.click(grade_pa, [a_audio, tgt], [a_md, a_tbl])

    return demo


demo = build()

if __name__ == "__main__":
    import sys
    print("slplab 음소 모델 로딩 준비(첫 채점 시 자동 다운로드/로드)...")
    # 큐 활성화: 긴 녹음 채점(수십 초)이 타임아웃으로 끊겨 '결과 공백'이 되지 않게.
    demo.queue(default_concurrency_limit=1).launch(share="--share" in sys.argv)
