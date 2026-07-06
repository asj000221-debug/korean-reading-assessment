"""난독증 임상 진단·처방 통합 데모 웹앱 (Gradio, 모델 불필요).

임상 계층(error_taxonomy / skill_map / phon_awareness / learner_profile / diagnose)을
한 화면에서 굴려보는 데모. ASR/torch 없이 텍스트 입력(dry-run)으로만 돌아서 어디서든
바로 뜬다(실음성 채점은 webapp.py). 탭은 단어 진단 / 회기 진단 리포트 / 발달 배치·처방
/ 음운인식 과제 / 종단 프로파일.

python clinical_app.py → 127.0.0.1:7861 (GRADIO_SHARE=1이면 공개 링크).
"""

import json
import os

import gradio as gr

from error_taxonomy import classify_word, summarize_profile
from skill_map import placement, prescribe, SKILLS, format_ladder
from phon_awareness import make_task, score_response
from diagnose import diagnose_session, format_report
import learner_profile as lp

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "item_bank.json"), encoding="utf-8") as f:
    BANK = json.load(f)

# 관찰된 오류쌍 프리셋(단어 진단 탭). 키는 드롭다운 라벨.
PPT_ERROR_EXAMPLES = {
    "가죽 → 아죽 (초성 생략, 사례 관찰)": ("가죽", "아죽"),
    "가방 → 바강 (음절 도치, 사례 관찰)": ("가방", "바강"),
    "자두 → 사두 (ㅈ→ㅅ 치찰음, 사례 관찰)": ("자두", "사두"),
    "진달래 → 신달래 (ㅈ→ㅅ 치찰음, 사례 관찰)": ("진달래", "신달래"),
    "야구 → 아구 (활음 단순화, 사례 관찰)": ("야구", "아구"),
    "카 → 콰 (활음 첨가, 사례 관찰)": ("카", "콰"),
    "웨이터 → 워이터 (w계 모음 혼동, 사례 관찰)": ("웨이터", "워이터"),
    "눈꺼풀 → 눈떠풀 (ㄲ→ㄸ, 사례 관찰)": ("눈꺼풀", "눈떠풀"),
    "멀튼 → 머튼 (종성 생략, 사례 관찰)": ("멀튼", "머튼"),
    "국물 → 국물 (비음화 미적용, 사례 관찰)": ("궁물", "국물"),
}

_SEV_EMOJI = {"high": "🔴", "med": "🟠", "low": "🟡"}


# ════════════════════════════════════════════════════════════════════
# 탭 1) 단어 진단
# ════════════════════════════════════════════════════════════════════
def diagnose_word(preset, expected, actual):
    if preset and preset in PPT_ERROR_EXAMPLES:
        expected, actual = PPT_ERROR_EXAMPLES[preset]
    expected, actual = (expected or "").strip(), (actual or "").strip()
    if not expected or not actual:
        return "기대 발음과 실제 발화를 입력하거나 PPT 예시를 선택하세요.", []
    ces = classify_word(expected, actual)
    if not ces:
        return f"✅ 「{expected}」 = 「{actual}」 — 오류 없음(정확히 읽음).", []
    rows = []
    for c in ces:
        label = SKILLS[c.skill].label if c.skill in SKILLS else c.skill
        rows.append([f"{_SEV_EMOJI.get(c.severity,'')} {c.pattern}", c.detail,
                     c.role, label, c.severity])
    head = f"「{expected}」 → 「{actual}」  :  임상 오류 {len(ces)}건 검출"
    return head, rows


def preset_fill(preset):
    if preset in PPT_ERROR_EXAMPLES:
        e, a = PPT_ERROR_EXAMPLES[preset]
        return e, a
    return gr.update(), gr.update()


# ════════════════════════════════════════════════════════════════════
# 탭 2) 회기 진단 리포트
# ════════════════════════════════════════════════════════════════════
# 데모 회기(사례연구 초·중기 오류 반영). 컬럼: 단어,기대발음,실제발화,유형,스킬,규칙
DEMO_SESSION_ROWS = [
    ["고기", "고기", "고기", "transparent", "decode_vowel_simple", ""],
    ["거미", "거미", "고미", "transparent", "decode_vowel_simple", ""],
    ["야구", "야구", "아구", "transparent", "decode_vowel_glide", ""],
    ["교회", "교회", "고회", "transparent", "decode_vowel_glide", ""],
    ["자두", "자두", "사두", "transparent", "percept_sibilant", ""],
    ["가죽", "가죽", "아죽", "transparent", "pa_phoneme", ""],
    ["국물", "궁물", "국물", "phonrule", "rule_nasal", "비음화"],
]


def run_session(rows):
    results = []
    for r in rows:
        if not r or not str(r[0]).strip():
            continue
        word, exp, act, itype = (str(r[0]).strip(), str(r[1]).strip(),
                                 str(r[2]).strip(), str(r[3]).strip() or "transparent")
        skill = str(r[4]).strip() if len(r) > 4 else ""
        rule = str(r[5]).strip() if len(r) > 5 else ""
        item = {"word": word, "expected": exp or word, "actual": act,
                "item_type": itype}
        if skill:
            item["skill"] = skill
        if rule:
            item["rule"] = rule
        results.append(item)
    if not results:
        return "행을 입력하세요(단어/기대발음/실제발화 필수)."
    dx = diagnose_session(results)
    return format_report(dx)


# ════════════════════════════════════════════════════════════════════
# 탭 3) 발달 배치/처방
# ════════════════════════════════════════════════════════════════════
# 배치 탭에서 다룰 대표 스킬(슬라이더). 전체 27개 중 핵심 라인.
LADDER_SKILLS = [
    "pa_syllable", "percept_consonant", "percept_sibilant", "pa_phoneme",
    "decode_vowel_simple", "decode_vowel_glide", "decode_vowel_diphthong",
    "decode_onset", "decode_coda", "rule_final7", "decode_nonword",
]


def run_placement(*vals):
    # vals: LADDER_SKILLS 순서의 정확도(0~100). 음수/미사용은 제외.
    evidence = {}
    for sid, v in zip(LADDER_SKILLS, vals):
        if v is not None and v >= 0:
            evidence[sid] = v / 100.0
    if not evidence:
        return "스킬 정확도를 하나 이상 설정하세요(−1 = 미검사).", ""
    pl = placement(evidence)
    ladder = format_ladder(evidence)
    # 약점 = 정확도 낮은 순(가중치 흉내: (1-acc))
    weak = sorted(((sid, (1 - a)) for sid, a in evidence.items() if a < 0.9),
                  key=lambda kv: -kv[1])
    rx = prescribe([(s, w) for s, w in weak], evidence)
    head = [f"▶ 현재 발달 단계: {pl.current_label}",
            "→ 다음 목표: " + (", ".join(SKILLS[t].label for t in pl.next_targets)
                            or "—"), "", ladder]
    pres = ["=== 개선방향(처방) ==="]
    for p in rx[:5]:
        pres.append(f"· [{p['domain']}] {p['label']}  ({p['ppt']})")
        pres.append(f"    → {p['method']}")
        if p["note"]:
            pres.append(f"    ⚠ {p['note']}")
    return "\n".join(head), "\n".join(pres)


# ════════════════════════════════════════════════════════════════════
# 탭 4) 음운인식 과제
# ════════════════════════════════════════════════════════════════════
PA_OPS = {
    "음절-수세기 (자동차→3)": ("syllable", "count", {}),
    "음절-합성 (연·필→연필)": ("syllable", "blend", {"parts": True}),
    "음절-분절 (전화→전 화)": ("syllable", "segment", {}),
    "음절-생략 (축구−구→축)": ("syllable", "delete", {"target": True}),
    "음절-첨가 (바지+청→청바지)": ("syllable", "add", {"syl": True}),
    "음절-도치 (나무→무나)": ("syllable", "reverse", {}),
    "음절-대치 (호박,호→수→수박)": ("syllable", "substitute", {"old": True, "new": True}),
    "음소-수세기 (강→3)": ("phoneme", "count", {}),
    "음소-합성 (ㅂ+ㅓ→버)": ("phoneme", "blend", {"jamos": True}),
    "음소-생략(첫소리) (기→이)": ("phoneme", "delete", {}),
    "음소-대치(첫소리) (저→ㅊ→처)": ("phoneme", "substitute", {"new_onset": True}),
}


def make_pa_task(op_label, stimulus, arg1, arg2):
    level, op, needs = PA_OPS[op_label]
    kw = {}
    a1, a2 = (arg1 or "").strip(), (arg2 or "").strip()
    if "parts" in needs:
        kw["parts"] = [p for p in a1.replace(",", " ").split() if p]
    if "jamos" in needs:
        kw["jamos"] = [p for p in a1.replace(",", " ").split() if p]
    if "target" in needs:
        kw["target"] = a1
    if "syl" in needs:
        kw["syl"], kw["pos"] = a1, "front"
    if "old" in needs:
        kw["old"], kw["new"] = a1, a2
    if "new_onset" in needs:
        kw["new_onset"] = a1
    try:
        t = make_task(level, op, stimulus.strip(), **kw)
    except Exception as e:
        return f"입력 확인 필요: {e}", ""
    return t.prompt, t.expected


def score_pa(op_label, stimulus, arg1, arg2, response):
    prompt, expected = make_pa_task(op_label, stimulus, arg1, arg2)
    if not expected:
        return prompt
    level, op, needs = PA_OPS[op_label]
    kw = {}
    a1, a2 = (arg1 or "").strip(), (arg2 or "").strip()
    if "parts" in needs:
        kw["parts"] = [p for p in a1.replace(",", " ").split() if p]
    if "jamos" in needs:
        kw["jamos"] = [p for p in a1.replace(",", " ").split() if p]
    if "target" in needs:
        kw["target"] = a1
    if "syl" in needs:
        kw["syl"], kw["pos"] = a1, "front"
    if "old" in needs:
        kw["old"], kw["new"] = a1, a2
    if "new_onset" in needs:
        kw["new_onset"] = a1
    t = make_task(level, op, stimulus.strip(), **kw)
    r = score_response(t, response)
    mark = "✅ 정답" if r["correct"] else "❌ 오류"
    skill = SKILLS[t.skill].label if t.skill in SKILLS else t.skill
    return (f"지시: {t.prompt}\n기대 응답: 「{r['expected']}」\n"
            f"아동 응답: 「{r['response']}」\n채점: {mark}  (정확도 {r['accuracy']:.0%})\n"
            f"측정 스킬: {skill}")


# ════════════════════════════════════════════════════════════════════
# 탭 5) 종단 프로파일
# ════════════════════════════════════════════════════════════════════
def build_profile(rows, learner_id):
    lid = (learner_id or "_webdemo").strip()
    path = lp._path(lid)
    if os.path.exists(path):
        os.remove(path)
    for r in rows:
        if not r or r[0] in (None, "") or str(r[0]).strip() == "":
            continue
        try:
            sess_no = int(float(r[0]))
        except Exception:
            continue
        date = str(r[1]).strip() if len(r) > 1 and r[1] else None
        skill_scores = {}
        if len(r) > 2 and str(r[2]).strip():
            for kv in str(r[2]).split(","):
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    try:
                        skill_scores[k.strip()] = float(v) / 100.0 if float(v) > 1 else float(v)
                    except Exception:
                        pass
        patterns = {}
        if len(r) > 3 and str(r[3]).strip():
            for kv in str(r[3]).split(","):
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    try:
                        patterns[k.strip()] = int(float(v))
                    except Exception:
                        pass
        fluency = {}
        if len(r) > 4 and str(r[4]).strip():
            try:
                fluency = {"eojeol_per_min": float(r[4])}
            except Exception:
                pass
        lp.append_session(lid, sess_no, skill_scores, patterns, fluency, date)
    prof = lp.load(lid)
    if not prof["sessions"]:
        return "회기 행을 입력하세요(회기번호 필수)."
    return lp.progress_note(prof)


DEMO_PROFILE_ROWS = [
    [9, "2016-09-01", "decode_vowel_simple:60, pa_syllable:80",
     "vowel_substitution:3, syllable_transposition:2", 12],
    [22, "2016-12-01", "decode_vowel_simple:85, decode_vowel_glide:50, pa_syllable:90",
     "glide_simplification:4", 18],
    [55, "2018-03-01",
     "decode_vowel_simple:95, decode_vowel_glide:90, decode_onset:90, decode_coda:85",
     "coda_deletion:2, sibilant_confusion:3", 30],
]

SKILL_IDS_HINT = "스킬 id: " + ", ".join(LADDER_SKILLS)

INTRO_MD = """
# 🧠 난독증 임상 진단·처방 — 통합 데모

익명화된 난독증 치료 사례연구(152회기)를 근거로, 기존 **해독 정확도 채점기**를
**진단·처방 엔진**으로 확장한 임상 계층을 한 화면에서 테스트합니다.
이 데모는 **모델 없이 텍스트 입력**으로 동작합니다(마이크 실음성 채점은 `webapp.py`).

| 탭 | 무엇을 하나 |
|---|---|
| **1 단어 진단** | 기대발음→실제발화 한 쌍을 **임상 오류유형**으로 분류 (PPT 실오류 예시 내장) |
| **2 회기 진단 리포트** | 여러 문항 채점 → 오류프로파일 → 약점스킬 → **발달 배치** → **개선방향 처방** |
| **3 발달 배치/처방** | 스킬별 정확도로 **발달 사다리** + 다음 목표 + 중재법 |
| **4 음운인식 과제** | 음절/음소 조작 과제 **생성 + 응답 채점** (글자 없는 청각과제) |
| **5 종단 프로파일** | 회기 누적 → **학습결과(긍정/한계)·유창성 추이** 자동 생성 |

> 전부 학습 0(규칙). 오류유형·발달위계·중재법·문항이 모두 실제 임상기록(PPT)에 근거합니다.
"""


# ════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════
with gr.Blocks(title="난독증 임상 진단·처방 데모") as demo:
    gr.Markdown("## 🧠 난독증 임상 진단·처방 — 통합 데모 (모델 불필요)")
    with gr.Tabs():
        # 1) 단어 진단
        with gr.Tab("1 단어 진단"):
            gr.Markdown("기대 발음과 실제 발화 한 쌍의 **임상 오류유형**을 분류합니다. "
                        "PPT 실관찰 오류 예시를 골라 바로 확인해보세요.")
            w_preset = gr.Dropdown(["(직접 입력)"] + list(PPT_ERROR_EXAMPLES),
                                   label="PPT 실오류 예시", value=list(PPT_ERROR_EXAMPLES)[0])
            with gr.Row():
                w_exp = gr.Textbox(label="기대 발음(G2P)", value="가죽")
                w_act = gr.Textbox(label="실제 발화(아동)", value="아죽")
            w_btn = gr.Button("진단", variant="primary")
            w_head = gr.Textbox(label="결과", lines=1)
            w_tbl = gr.Dataframe(headers=["오류유형", "상세", "위치", "약점 하위기술", "심각도"],
                                 label="임상 오류", wrap=True)
            w_preset.change(preset_fill, w_preset, [w_exp, w_act])
            w_btn.click(diagnose_word, [w_preset, w_exp, w_act], [w_head, w_tbl])

        # 2) 회기 진단 리포트
        with gr.Tab("2 회기 진단 리포트"):
            gr.Markdown("한 회기에 읽은 문항들을 표로 입력하고 **진단 리포트**를 생성합니다. "
                        "컬럼: 단어 / 기대발음 / 실제발화 / 유형(transparent·phonrule·nonword) / "
                        "스킬id(선택) / 규칙(phonrule 시).\n\n" + SKILL_IDS_HINT)
            s_tbl = gr.Dataframe(
                headers=["단어", "기대발음", "실제발화", "유형", "스킬id", "규칙"],
                value=DEMO_SESSION_ROWS, row_count=(7, "dynamic"),
                col_count=(6, "fixed"), wrap=True, label="회기 문항")
            s_btn = gr.Button("진단 리포트 생성", variant="primary")
            s_out = gr.Textbox(label="임상 진단 리포트", lines=28)
            s_btn.click(run_session, s_tbl, s_out)

        # 3) 발달 배치/처방
        with gr.Tab("3 발달 배치/처방"):
            gr.Markdown("스킬별 **정확도(%)**를 설정하면 발달 사다리 위 위치와 다음 목표·"
                        "개선방향을 산출합니다. (−1 = 미검사)")
            sliders = []
            for sid in LADDER_SKILLS:
                sliders.append(gr.Slider(-1, 100, value=-1, step=5,
                                         label=f"{SKILLS[sid].label}  [{sid}]"))
            p_btn = gr.Button("배치·처방", variant="primary")
            with gr.Row():
                p_ladder = gr.Textbox(label="발달 배치", lines=22)
                p_presc = gr.Textbox(label="개선방향(처방)", lines=22)
            p_btn.click(run_placement, sliders, [p_ladder, p_presc])

        # 4) 음운인식 과제
        with gr.Tab("4 음운인식 과제"):
            gr.Markdown("글자 없이 소리만 조작하는 **음운인식 과제**를 생성하고 아동 응답을 "
                        "채점합니다. 인자칸: 합성=음절/자모(공백구분), 생략/대치=대상.")
            pa_op = gr.Dropdown(list(PA_OPS), label="과제 유형", value=list(PA_OPS)[5])
            with gr.Row():
                pa_stim = gr.Textbox(label="자극(stimulus)", value="나무")
                pa_a1 = gr.Textbox(label="인자1(합성요소/대상/새소리)", value="")
                pa_a2 = gr.Textbox(label="인자2(대치 새음절)", value="")
            pa_make = gr.Button("과제 생성(기대응답 보기)")
            pa_prompt = gr.Textbox(label="지시문", lines=1)
            pa_expected = gr.Textbox(label="기대 응답", lines=1)
            gr.Markdown("— 아동 응답을 입력해 채점 —")
            pa_resp = gr.Textbox(label="아동 응답", value="무나")
            pa_score = gr.Button("채점", variant="primary")
            pa_out = gr.Textbox(label="채점 결과", lines=6)
            pa_make.click(make_pa_task, [pa_op, pa_stim, pa_a1, pa_a2],
                          [pa_prompt, pa_expected])
            pa_score.click(score_pa, [pa_op, pa_stim, pa_a1, pa_a2, pa_resp], pa_out)

        # 5) 종단 프로파일
        with gr.Tab("5 종단 프로파일"):
            gr.Markdown("회기를 누적해 **학습결과(긍정/한계)·추이·유창성**을 자동 생성합니다. "
                        "컬럼: 회기번호 / 날짜 / 스킬정확도(id:%, 쉼표구분) / "
                        "오류유형(pattern:횟수) / 유창성(어절/분).")
            pr_id = gr.Textbox(label="학습자 id", value="_webdemo")
            pr_tbl = gr.Dataframe(
                headers=["회기", "날짜", "스킬정확도", "오류유형", "유창성"],
                value=DEMO_PROFILE_ROWS, row_count=(3, "dynamic"),
                col_count=(5, "fixed"), wrap=True, label="회기 기록")
            pr_btn = gr.Button("프로파일 생성", variant="primary")
            pr_out = gr.Textbox(label="학습결과 요약(자동)", lines=10)
            pr_btn.click(build_profile, [pr_tbl, pr_id], pr_out)

        with gr.Tab("ℹ️ 설명"):
            gr.Markdown(INTRO_MD)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_PORT", "7861")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
    )
