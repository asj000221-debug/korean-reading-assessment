"""HuggingFace Spaces 진입점.

webapp.py에서 만든 Gradio `demo`를 그대로 구동한다.
Spaces가 호스트/포트를 자동 설정하므로 share 불필요.
모델(slplab 음소, kresnik)은 첫 실행 시 HF Hub에서 자동 다운로드된다.
"""

# --- gradio_client 알려진 버그 패치 ---
# API 스키마에 additionalProperties=bool 이 있으면 json_schema_to_python_type가
# 'argument of type bool is not iterable'로 죽는다("no api found"의 원인).
# 스키마가 bool이면 안전하게 처리하도록 감싼다.
import gradio_client.utils as _gcu  # noqa: E402

_orig_jstpt = _gcu._json_schema_to_python_type


def _safe_jstpt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_jstpt(schema, defs)


_gcu._json_schema_to_python_type = _safe_jstpt

from webapp import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch()
