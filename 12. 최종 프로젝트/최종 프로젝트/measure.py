"""[3단계] 웨이퍼 식각 측정값 — settings.CSV_FILE(Zenodo 17122442, CC BY 4.0, 원본 그대로, 첨부자료로 별도 배포)

열: experiment_key 실험 ID · lot_number · wafer_number · loc_id 측정 위치(실험마다 9개 점) · X, Y 좌표 ·
    preox_thickness · postox_thickness · stepheight · oxide_etch 산화막 식각량 · si_etch 실리콘 식각량
load_rows(experiment_key: str) -> list[dict]   그 실험의 행(숫자 열은 float), 없으면 []
compare_etch_locations(experiment_key: str) -> str
    @tool: si_etch·oxide_etch 각각 최댓값 위치와 값을 찾고 record.add_source로 측정 카드 등록 →
    "실험 …(측정점 9개)에서 si_etch 최댓값 위치는 …(값), oxide_etch 최댓값 위치는 …(값)로 같습니다/다릅니다. [n]"
    없는 실험 ID → "조회 결과 없음: 실험 ID …는 … 에 없습니다." (출처 카드 없음)
run_pandas_code(code: str) -> str
    @tool: 측정 데이터 df에 pandas 코드를 실행해 결과 문자열을 돌려준다(실행 오류도 문자열로 안내)
build_data_agent()                               create_agent(채팅 모델, [run_pandas_code], 지시문)
    지시문: 답 전에 pandas 코드로 df를 확인하고 근거 수치를 답에 넣는다. 업로드된 데이터로 확인할 수 없는
    질문은 추측하지 않고 "제공된 데이터로는 확인할 수 없습니다."로 답변을 제한한다
"""
import io
from contextlib import redirect_stdout

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool

import settings
from llm import get_chat_model
from record import add_source

DATA_URL = "https://zenodo.org/records/17122442"
NUMERIC_COLUMNS = ("lot_number", "wafer_number", "X", "Y", "preox_thickness", "postox_thickness", "stepheight", "oxide_etch", "si_etch")
TOOL_RESULT_LIMIT = 2000  # pandas 실행 결과가 길면 잘라 토큰을 아낀다

DATA_PROMPT = """너는 측정 데이터 분석 담당이다. run_pandas_code 도구로 측정 데이터 df에 pandas 코드를 실행해 답을 확인한다.
- df: Si_Oxide_etch_9_points.csv — 열: experiment_key(실험 ID, 예: 2024-07-02_01), loc_id(측정 위치), X, Y,
  si_etch(실리콘 식각량), oxide_etch(산화막 식각량), stepheight 등. 실험마다 9개 측정점.
- 답하기 전에 반드시 run_pandas_code로 데이터를 확인하고, 결과는 print()로 출력해 확인한다.
- 답에는 근거가 되는 수치(실험 ID·위치·값)를 포함한다.
- 업로드된 데이터(df)로 확인할 수 없는 부분은 추측이나 외부 지식으로 답하지 말고
  "제공된 데이터로는 확인할 수 없습니다."로 답한다."""


def _df() -> "pd.DataFrame":
    df = pd.read_csv(settings.CSV_FILE)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def load_rows(experiment_key: str) -> list:
    rows = _df()[_df()["experiment_key"] == experiment_key]
    return rows.to_dict("records")


@tool
def compare_etch_locations(experiment_key: str) -> str:
    """실험 ID로 측정 데이터를 조회해 si_etch(실리콘 식각량)와 oxide_etch(산화막 식각량)의 최댓값 위치와 값을 비교한다. 실험 ID 형식 예: 2024-07-02_01"""
    rows = load_rows(experiment_key)
    if not rows:
        return f"조회 결과 없음: 실험 ID {experiment_key}는 측정 데이터에 없습니다."
    si_max = max(rows, key=lambda r: r["si_etch"])
    oxide_max = max(rows, key=lambda r: r["oxide_etch"])
    card = {
        "kind": "measure",
        "dataset": settings.CSV_FILE.name,
        "experiment_key": experiment_key,
        "columns": ["si_etch", "oxide_etch"],
        "points": len(rows),
        "si_max_loc": si_max["loc_id"],
        "si_max": si_max["si_etch"],
        "oxide_max_loc": oxide_max["loc_id"],
        "oxide_max": oxide_max["oxide_etch"],
        "table": [{"loc_id": r["loc_id"], "si_etch": r["si_etch"], "oxide_etch": r["oxide_etch"]} for r in rows],
        "url": DATA_URL,
    }
    n = add_source(card, key=f"measure:{experiment_key}")
    same = "같습니다" if si_max["loc_id"] == oxide_max["loc_id"] else "다릅니다"
    return (
        f"실험 {experiment_key}(측정점 {len(rows)}개)에서 si_etch 최댓값 위치는 {si_max['loc_id']}({si_max['si_etch']}), "
        f"oxide_etch 최댓값 위치는 {oxide_max['loc_id']}({oxide_max['oxide_etch']})로 {same}. [{n}]"
    )


@tool
def run_pandas_code(code: str) -> str:
    """측정 데이터 df(experiment_key, loc_id, X, Y, si_etch, oxide_etch 등 열)에 실행할 pandas 코드를 실행하고 출력을 돌려준다. 결과는 print()로 출력한다."""
    ns = {"df": _df(), "pd": pd}
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            try:
                result = eval(code, ns)  # 한 줄 식이면 값을 바로 돌려준다
            except SyntaxError:
                exec(code, ns)  # 여러 문이면 실행하고 print 출력만 모은다
                result = None
        output = buffer.getvalue()
        if result is not None:
            output += str(result)
        output = output.strip()
        if not output:
            return "실행 결과 없음. print()로 결과를 출력해 주세요."
        if len(output) > TOOL_RESULT_LIMIT:
            output = output[:TOOL_RESULT_LIMIT] + " …(결과가 길어 잘림)"
        return output
    except Exception as e:
        return f"코드 오류: {type(e).__name__}: {e}"


def build_data_agent():
    return create_agent(get_chat_model(), [run_pandas_code], system_prompt=DATA_PROMPT)
