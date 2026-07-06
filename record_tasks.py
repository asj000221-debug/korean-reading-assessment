# -*- coding: utf-8 -*-
"""3과제 라이브 녹음·채점 하네스 (slplab CTC).

run_tasks.py 채점 로직을 '직접 말해서' 확인하는 대화형 도구. 마이크는 사용자 본인
터미널에서만 잡혀서(에이전트 프로세스엔 오디오 세션이 없다) 사용자가 직접 돌린다.

    python record_tasks.py words [--section meaning|nonsense|all]
    python record_tasks.py para
    python record_tasks.py pa [--targets 따,규,기떠]   # 생략 시 16개 전부
    python record_tasks.py --list-devices

[Enter]로 녹음 시작·정지 → 음소 인식 → 채점. 원본은 recordings/에 타임스탬프로
남겨 run_tasks --wav 로 재채점할 수 있다. 낱말/단락은 열린 전사, 음운인식은
닫힌집합 forced-align — 과제별 패러다임을 그대로 따른다.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

import task_db

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "recordings")


# ══════════════ 마이크 / 녹음 ══════════════
def pick_input_device(device):
    if device is not None:
        return device
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            return default_in
    except Exception:
        pass
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            return i
    raise RuntimeError("입력(마이크) 장치를 찾지 못했습니다. --list-devices 로 확인하세요.")


def list_devices():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i:3}  {d['name']}  (in {d['max_input_channels']}ch)")


def record_until_enter(device, countdown=True, max_seconds=180):
    """[Enter]로 시작→정지하는 가변길이 녹음 → 16kHz mono float32 numpy.

    긴 낱말목록/단락도 시간제한 없이 편하게 읽도록 press-to-stop 방식.
    """
    if countdown:
        print("   준비... ", end="", flush=True)
        for n in (3, 2, 1):
            print(f"{n} ", end="", flush=True)
            sd.sleep(600)
    print("🎤 녹음 중 — 다 읽으면 [Enter]", flush=True)

    frames = []

    def cb(indata, _n, _t, status):
        if status:
            pass  # over/underflow는 무시(정확도에 큰 영향 없음)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            callback=cb, device=device)
    with stream:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass

    if not frames:
        return np.zeros(0, dtype="float32")
    audio = np.concatenate(frames).reshape(-1)[: int(max_seconds * SR)]
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    dur = len(audio) / SR
    print(f"   (녹음 {dur:.1f}s, 최대진폭 {peak:.3f})")
    if peak < 0.01:
        print("   ⚠ 소리가 거의 없습니다. 마이크/장치 번호(--list-devices)를 확인하세요.")
    return audio


def save_wav(audio, tag):
    os.makedirs(REC_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join(REC_DIR, f"live_{tag}_{stamp}.wav")
    sf.write(path, audio, SR)
    return path


# ══════════════ ① 낱말읽기 ══════════════
def run_words(device, section):
    from phoneme_asr import transcribe_phones
    from phoneme_map import phones_to_hangul
    from run_tasks import score_words, report_words

    items = task_db.word_items(section)
    words = [it["word"] for it in items]
    print(f"\n▶ 낱말읽기 [{section}] — {len(words)}개 낱말을 위→아래 순서대로 또박또박 읽으세요.\n")
    # 10개씩 줄바꿈해서 제시
    for i in range(0, len(words), 10):
        print("   " + "  ".join(f"{w}" for w in words[i:i + 10]))
    input("\n   [Enter] 누르면 녹음 시작...")
    audio = record_until_enter(device)
    if len(audio) == 0:
        print("녹음 실패."); return
    path = save_wav(audio, f"words_{section}")

    print("   음소 인식(slplab) 중...")
    actual = transcribe_phones(path)
    print(f"\n들린 전체: 「{phones_to_hangul(actual)}」\n")
    its, scores, heard = score_words(actual, section)
    print(report_words(its, scores, heard))
    print(f"\n원본: {path}\n재채점: python run_tasks.py words --wav {os.path.relpath(path, HERE)} --section {section}")


# ══════════════ ② 단락읽기 ══════════════
def run_para(device):
    from phoneme_asr import transcribe_phones
    from run_tasks import score_paragraph, report_paragraph

    p = task_db.paragraph()
    print(f"\n▶ 단락읽기 [{p['id']}] — 아래 글을 소리내어 읽으세요 (약 {p['n_eojeol']}어절).\n")
    print(p["text"])
    input("\n   [Enter] 누르면 녹음 시작...")
    audio = record_until_enter(device)
    if len(audio) == 0:
        print("녹음 실패."); return
    path = save_wav(audio, "para")

    print("   음소 인식(slplab) 중...")
    actual = transcribe_phones(path)
    surf, scores, heard = score_paragraph(actual)
    print("\n" + report_paragraph(surf, scores))
    print(f"\n원본: {path}\n재채점: python run_tasks.py para --wav {os.path.relpath(path, HERE)}")


# ══════════════ ③ 음운인식(합성) ══════════════
def run_pa(device, targets):
    import pa_synth

    items = {it["target"]: it for it in task_db.pa_items()}
    if targets:
        order = [t.strip() for t in targets if t.strip() in items]
        missing = [t for t in targets if t.strip() and t.strip() not in items]
        if missing:
            print(f"⚠ 후보에 없는 타깃 무시: {missing}  (후보={task_db.pa_candidates()})")
    else:
        order = list(items.keys())
    if not order:
        print("진행할 타깃이 없습니다."); return

    print(f"\n▶ 음운인식(합성) — {len(order)}개 타깃. 각 화면의 소리를 '합쳐서' 한 번에 발음하세요.")
    print("  (검사자가 음소를 끊어 들려주고, 아동이 합성해 말하는 과제)\n")

    results = []
    for idx, tgt in enumerate(order, 1):
        it = items[tgt]
        print(f"\n[{idx}/{len(order)}] 목표 「{tgt}」 ({it['n_phone']}음소, {it['bin']})")
        input("   [Enter] 누르면 녹음 시작...")
        audio = record_until_enter(device, countdown=False)
        if len(audio) == 0:
            print("   (녹음 없음 — 건너뜀)")
            continue
        path = save_wav(audio, f"pa_{tgt}")
        print("   닫힌집합 forced-align 중...")
        v = pa_synth.verify(path, target=tgt)
        mark = "✅" if v["correct"] else "❌"
        print(f"   {mark} pred={v['pred']} correct={v['correct']} "
              f"reject={v.get('reject')} min_gop={v.get('min_gop')}")
        print(f"      순위: {v['ranking']}")
        results.append({"item": it, "verify": v})

    if results:
        agg = pa_synth.score_all(results)
        print("\n── 음운인식 정답률 ──")
        for k, v in agg.items():
            print(f"  {k}: {v['correct']}/{v['total']} ({v['rate']:.0%})")


# ══════════════ 진입점 ══════════════
def main():
    ap = argparse.ArgumentParser(description="매뉴얼 3과제 라이브 녹음·채점")
    ap.add_argument("--device", type=int, default=None, help="마이크 장치 번호")
    ap.add_argument("--list-devices", action="store_true")
    sub = ap.add_subparsers(dest="task")

    pw = sub.add_parser("words", help="① 낱말읽기")
    pw.add_argument("--section", default="all", choices=["all", "meaning", "nonsense"])
    sub.add_parser("para", help="② 단락읽기")
    pp = sub.add_parser("pa", help="③ 음운인식(합성)")
    pp.add_argument("--targets", default="", help="쉼표구분 타깃(생략=16개 전부)")

    args = ap.parse_args()

    if args.list_devices:
        list_devices(); return
    if not args.task:
        ap.print_help(); return

    device = pick_input_device(args.device)
    print(f"사용 마이크: {device} ({sd.query_devices(device)['name']})")
    print("slplab 음소 모델 로딩 중(최초 1회)...")
    from phoneme_asr import _load
    _load()

    if args.task == "words":
        run_words(device, args.section)
    elif args.task == "para":
        run_para(device)
    elif args.task == "pa":
        run_pa(device, args.targets.split(",") if args.targets else [])


if __name__ == "__main__":
    main()
