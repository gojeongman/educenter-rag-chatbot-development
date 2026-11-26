"""시험 함수(제공, 고치지 않음) — 1일차 주피터 실습의 채점 함수를 글자 그대로 옮겼습니다.
화면에는 각 함수가 출력하는 ✅/❌ 줄을 그대로 보여 줍니다.
"""
import contextlib
import io

shift_agent = None   # 서버가 연속 시험 직전에 chatbot.AGENT로 바꿉니다
config_for = None    # 서버가 연속 시험 직전에 chatbot.config_for로 바꿉니다

# ✅ 채점 셀 — 한 번의 호출(턴) 검사 함수 (고치지 마세요)
import re

def _to_number(value):
    """'60', '60.0', '1,200', '-60'을 실제 숫자값으로 바꿉니다. 소수부를 버리지 않습니다. 숫자가 아니면 None."""
    try:
        return float(str(value).replace(",", "").replace("−", "-").strip())
    except (TypeError, ValueError):
        return None

def _show(number):
    return int(number) if number == number and float(number).is_integer() else number

def answer_quantities(text):
    """최종 답변에서 단위 '개'가 붙은 수량을 부호·소수부까지 읽어 실제 숫자값으로 돌려줍니다.
    '60개'·'60 개'·'60.0개' → 60, '60.9개' → 60.9, '-60개' → -60, '160개' → 160
    '시간당 240개', '240개/시간'처럼 시간당 생산량을 말하는 숫자는 결과 수량에서 뺍니다.
    '50~60개'·'50-60개'처럼 범위로 말한 수량은 하나의 값이 아니므로 판정할 수 없는 값(nan)으로 둡니다."""
    text = str(text)
    found = []
    pattern = r"(?<![\d.,])([-−+]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*개"
    for m in re.finditer(pattern, text):
        before = text[max(0, m.start() - 12):m.start()]
        after = text[m.end():m.end() + 3].lstrip()
        if "시간당" in before or "시간에" in before or after.startswith(("/", "씩")):
            continue
        sign, whole, frac = m.group(1), m.group(2), m.group(3) or ""
        prev = text[m.start() - 1] if m.start() > 0 else ""
        if not sign and prev and prev in "-−~～":  # 범위의 끝값 (답이 숫자로 시작하면 prev는 빈 문자열)
            found.append(float("nan"))
            continue
        found.append(_show(_to_number(sign + whole + frac)))
    return found

def quantity_is(text, expected):
    """답변의 결과 수량(단위 '개')이 숫자값으로 expected 하나뿐인지 판정합니다."""
    qty = answer_quantities(text)
    return bool(qty) and all(q == expected for q in qty), qty

ASK_WORDS = ("?", "알려", "주세요", "주시겠", "필요합니다", "필요해요", "말씀해")
RATE_WORDS = ("시간당", "한 시간", "1시간", "시간에", "생산량", "생산 속도", "몇 개")

def check_turn(new_messages, expected_tools, expected_qty=None, ask_missing=False, question=""):
    """한 번의 호출(턴)을 채점합니다.
    반환값: True = 자동 판정 통과 / False = 실패 / None = 자동 판정 항목은 통과했지만 학생 확인 필요"""
    if not isinstance(new_messages, list):
        print("  ❌ ask_agent()가 '이번 호출에서 새로 생긴 메시지 목록'을 반환해야 합니다.")
        return False
    calls = [c for m in new_messages if m.type == "ai" for c in m.tool_calls]
    returns = [m for m in new_messages if m.type == "tool"]
    final = str(new_messages[-1].content) if new_messages else ""
    checks = {
        "이번 호출분만 반환·출력(사용자 메시지 1개)": sum(m.type == "human" for m in new_messages) == 1,
        f"이번 호출의 도구 실행 {len(expected_tools)}건": len(calls) == len(expected_tools) == len(returns),
    }
    for (name, args, value), call, ret in zip(expected_tools, calls, returns):
        checks[f"도구 {name} 호출"] = call["name"] == name
        checks[f"인자 {args}"] = all(_to_number(call["args"].get(k)) == v for k, v in args.items())
        checks[f"ToolMessage 반환값 {value}"] = _to_number(ret.content) == value
    if expected_qty is not None:
        ok, qty = quantity_is(final, expected_qty)
        checks[f"최종 답변의 수량이 {expected_qty}개 (찾은 수량: {qty or '없음'})"] = ok
    review = []
    if ask_missing:
        given = set(re.findall(r"\d+", question))
        scan = re.sub(r"1\s*시간", "", final)                 # '1시간에 몇 개' 같은 표현은 허용
        new_numbers = [n for n in re.findall(r"\d+", scan) if n not in given]
        qty = answer_quantities(final)
        checks[f"주어지지 않은 값을 단정하지 않음 (질문에 없던 숫자: {new_numbers or '없음'}, 결과 수량: {qty or '없음'})"] = not new_numbers and not qty
        if any(w in final for w in ASK_WORDS) and any(w in final for w in RATE_WORDS):
            checks["부족한 시간당 생산량을 되물음"] = True
        else:
            review.append("최종 답변이 '시간당 생산량'을 실제로 묻고 있는지 직접 읽고 판단하세요.")
    for label, ok in checks.items():
        print("  ✅" if ok else "  ❌", label)
    for item in review:
        print("  🔎 학생 확인:", item)
    if not all(checks.values()):
        return False
    return None if review else True

def count_saved(thread_id):
    return len(shift_agent.get_state(config_for(thread_id)).values.get("messages", []))

def check_loss_tool(calc):
    """1단계(TODO 1) 시험 — 모델·API를 부르지 않고 도구를 직접 실행한다.
    채점: @tool 도구 · 인자 이름 · 시간당 240개·정지 15분 → 60 · TOOLS에 두 도구. 도구 설명은 🔎 학생 확인."""
    import inspect
    import re
    names = [getattr(t, "name", repr(t)) for t in getattr(calc, "TOOLS", [])]
    loss = getattr(calc, "calculate_production_loss", None)
    if loss is None:
        print("❌ calculate_production_loss가 아직 없습니다 — [미구현] TODO 1 (baseline/calc_tools.py)")
        print(f"   현재 TOOLS: {names}")
        print("→ 1단계 미통과")
        return False
    results = []
    is_tool = hasattr(loss, "invoke") and hasattr(loss, "args") and hasattr(loss, "description")
    print(("✅" if is_tool else "❌") + " @tool로 정의한 도구 calculate_production_loss")
    results.append(is_tool)
    if is_tool:
        args = sorted(loss.args)
        ok = args == ["stopped_minutes", "units_per_hour"]
        print(("✅" if ok else "❌") + f" 인자 이름 {args} (units_per_hour, stopped_minutes)")
        results.append(ok)
        try:
            value = loss.invoke({"units_per_hour": 240, "stopped_minutes": 15})
        except Exception as e:
            value = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
        ok = value == 60 and not isinstance(value, bool)
        print(("✅" if ok else "❌") + f" 시간당 240개 · 정지 15분 → 반환값 {value!r} (기대 60)")
        results.append(ok)
    ok = "calculate_production" in names and "calculate_production_loss" in names
    print(("✅" if ok else "❌") + f" TOOLS 등록 {names} (두 도구 모두 있어야 함)")
    results.append(ok)
    if is_tool:
        desc = " ".join((loss.description or "").split())
        hint = "한국어 · 정지/멈춤 언급" if re.search("[가-힣]", desc) and re.search("정지|멈", desc) else "한국어로 정지(멈춘) 시간의 손실임을 적었는지 확인"
        print(f"🔎 학생 확인 — 도구 설명: {desc[:120]} ({hint})")
        try:
            reuse = "expected_output(" in inspect.getsource(loss.func)
        except (OSError, TypeError, AttributeError):
            reuse = False
        print("🔎 학생 확인 — " + ("expected_output을 다시 사용함" if reuse else "계산을 expected_output으로 했는지 확인"))
    passed = all(results)
    print("→ 1단계 통과" if passed else "→ 1단계 미통과 — ❌ 줄을 고치고 다시 시험하세요")
    return passed



def run_check(fn, *args, **kwargs):
    """시험 함수를 실행하고 출력 줄과 결과(True/False/None)를 돌려준다."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
    except Exception as e:  # 시험할 수 없는 상태(예: 기록 저장소 없음)도 화면에 이유를 보여 준다
        return {"result": False, "lines": [f"❌ 검사하지 못했습니다 — {type(e).__name__}: {e}"]}
    return {"result": result, "lines": [l for l in buf.getvalue().splitlines() if l.strip()]}


if __name__ == "__main__":
    # 1단계 직접 시험(모델·API 호출 없음): starter 폴더에서  python checks.py
    from baseline import calc_tools
    check_loss_tool(calc_tools)
