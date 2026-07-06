"""HF Spaces 배포 — 캐시된 huggingface-cli 로그인(또는 HF_TOKEN)을 사용.

사용:
    python deploy_to_hf.py <space-name>
예:
    python deploy_to_hf.py korean-reading-assessment

토큰은 코드에 넣지 않는다. `huggingface-cli login`으로 캐시된 인증을 자동 사용.
"""

import sys

from huggingface_hub import HfApi

UPLOAD_DIR = "."
IGNORE = [
    "recordings/*", "*.wav", "*.exe", "_*.py", "__pycache__/*", "*.pyc",
    "sample_responses.json", "make_dummy_wav.py", "debug_*.py", "compare.py",
    "explore_phoneme.py", "verify_model.py", "test_*.py", "record_and_score.py",
    "deploy_to_hf.py", ".gitignore",
]


def main():
    if len(sys.argv) < 2:
        print("usage: python deploy_to_hf.py <space-name>")
        raise SystemExit(1)
    name = sys.argv[1]

    api = HfApi()
    who = api.whoami()  # 캐시된 토큰 필요(없으면 여기서 에러)
    user = who["name"]
    repo_id = f"{user}/{name}"
    print(f"로그인 사용자: {user}")
    print(f"Space 생성/업로드: {repo_id}")

    api.create_repo(
        repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True
    )
    api.upload_folder(
        folder_path=UPLOAD_DIR,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="deploy korean reading assessment",
    )
    print(f"\n완료! → https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
