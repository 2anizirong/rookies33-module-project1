"""
agent.py

<필요 함수>
predict_price(), detect_anomaly(): predict_laptop.py 에서 import만 하기
web_search / file_search: 호스팅 툴로 등록
orchestrate(): 대화 오케스트레이션 (Function Calling 왕복 처리)

<이번 수정 사항 (최종본)>
1. [속도 개선] trim_conversation_history()로 최근 N턴만 유지, developer 프롬프트는 history에 안 쌓음.
2. [환율] get_usd_krw_rate()로 실제 환율을 가져와 시스템 프롬프트에 숫자로 박아넣음 (6시간 캐싱).
3. [출시연도] AI가 브랜드/CPU 등으로 스스로 추정 (무조건 -1 아님).
4. [버그 수정] response.output13.6 오타 -> response.output.
5. [출시연도만 자동 추정] brand/model_family/cpu_family/ram_gb/storage_type/storage_capacity_gb/
   screen_size_inch/os/condition 은 사용자가 말 안 하면 반드시 되물어야 함 (임의로 채우면 안 됨).
   딱 release_year(출시연도) 하나만 예외로, 사용자가 말 안 했으면 되묻지 않고 AI가 브랜드/CPU 등을
   근거로 스스로 추정해서 채움.
6. [web_search 남용 차단 - 프롬프트가 아니라 코드로 강제]
   팀원들이 프롬프트에 "검색 1~2번만 해" 라고 적어놨는데도 GPT가 그 규칙을 안 지키는 문제가 있었음.
   프롬프트 지시는 "권고"일 뿐 강제가 안 되므로, 아래 방식으로 코드 레벨에서 원천 차단함:
     (a) 사용자가 "시세/최근/동향/뉴스" 같은 키워드를 쓴 게 아니면 web_search 툴 자체를 요청에 안 넣음
         (옵션에 아예 없으면 GPT가 호출을 못 함)
     (b) 그래도 한 번 검색을 했으면, 같은 질문 처리 중에는 다음 라운드부터 web_search 툴을 목록에서 제거
         (한 번 쓰면 그걸로 끝, 재검색 물리적으로 불가능)
   file_search도 동일한 방식으로 처리.

<사전 준비>
pip install requests  (환율 API 호출용, 없으면 추가 설치 필요)
"""

# ==========================================================
# 모듈 import (전부 파일 최상단에 모음)
# ==========================================================
import json
import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from openai import OpenAI

# [경로 문제 해결] predict_laptop.py 안에서 "models/xxx.pkl" 같은 상대경로를 쓰고 있어서,
# 이 파일을 어느 폴더에서 실행하느냐에 따라 FileNotFoundError가 날 수 있었음.
# predict_laptop.py는 건드리지 않고, 여기서 작업 디렉토리를 이 agent.py 파일이 있는 폴더로
# 고정시켜서 상대경로가 항상 같은 곳을 가리키도록 함.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from predict_laptop import predict_price, detect_anomaly   # predict_laptop.py에서 가져오기

# [사전 점검] chdir을 해도 모델 파일이 진짜 없는 경우, 나중에 알 수 없는 traceback 대신
# 실행 초반에 바로 알아볼 수 있게 경로를 확인해준다.
# (실제 파일명은 predict_laptop.py의 joblib.load(...) 부분과 반드시 일치해야 함 -> price_model.pkl)
_EXPECTED_MODEL_PATH = os.path.join(os.getcwd(), "models", "price_model.pkl")
if not os.path.exists(_EXPECTED_MODEL_PATH):
    print(f"[경고] 모델 파일을 못 찾음: {_EXPECTED_MODEL_PATH}")
    print(f"[경고] 현재 작업 폴더: {os.getcwd()}")
    print(f"[경고] 이 폴더 안에 'models/price_model.pkl'이 실제로 있는지 확인해줘.")

# ==========================================================
# 상수 / 클라이언트
# ==========================================================

# .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
load_dotenv()

# OpenAI API Client 생성 - 환경변수에서 키를 불러와 사용 (하드코딩 금지)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# File Search에서 사용할 Vector Store ID (이것도 .env에서 관리하는 걸 추천)
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# 실제 함수 이름으로 함수 객체 매핑하기 (agent가 호출 요청하면 여기서 실행)
AVAILABLE_FUNCTIONS = {
    "predict_price": predict_price,
    "detect_anomaly": detect_anomaly
}

# 대화 히스토리를 얼마나 유지할지 (최근 사용자 턴 개수 기준)
KEEP_LAST_N_TURNS = 3

# 한 번의 질문(orchestrate 호출) 안에서 web_search / file_search를 최대 몇 번 허용할지
MAX_WEB_SEARCH_PER_QUESTION = 1
MAX_FILE_SEARCH_PER_QUESTION = 1

# 환율 캐시 (매 메시지마다 API 호출하면 느려지므로 일정 시간 캐싱)
_exchange_rate_cache = {"rate": None, "date": None, "fetched_at": None}
_CACHE_TTL = timedelta(hours=6)

# ==========================================================
# 환율 조회 (실시간, 캐싱 적용)
# ==========================================================
def get_usd_krw_rate() -> tuple:
    """
    오늘 기준 USD -> KRW 환율을 가져온다.
    - 6시간 이내에 이미 가져온 값이 있으면 그 값을 재사용 (속도 개선)
    - 1차: Frankfurter API (ECB 기준, 무료/키 불필요)
    - 2차: open.er-api.com (1차 실패 시 대체)
    - 둘 다 실패하면 고정값으로 폴백 (서비스가 죽지 않도록)
    반환값: (rate: float, rate_date: str)
    """
    now = datetime.now()
    cached_rate = _exchange_rate_cache["rate"]
    fetched_at = _exchange_rate_cache["fetched_at"]

    if cached_rate is not None and fetched_at is not None and (now - fetched_at) < _CACHE_TTL:
        return cached_rate, _exchange_rate_cache["date"]

    rate = None
    rate_date = None

    try:
        resp = requests.get("https://api.frankfurter.app/latest?from=USD&to=KRW", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"]["KRW"]
        rate_date = data["date"]
    except Exception as e:
        print(f"[경고] 환율 API(Frankfurter) 호출 실패: {e} -> 대체 API 시도")
        try:
            resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            rate = data["rates"]["KRW"]
            rate_date = data.get("time_last_update_utc", now.strftime("%Y-%m-%d"))
        except Exception as e2:
            print(f"[경고] 대체 환율 API도 실패: {e2} -> 고정 환율로 대체")
            rate = 1380.0
            rate_date = "고정값(환율 API 응답 실패)"

    _exchange_rate_cache.update({"rate": rate, "date": rate_date, "fetched_at": now})
    return rate, rate_date


# ==========================================================
# 대화 히스토리 트리밍 (속도 개선 핵심)
# ==========================================================
def trim_conversation_history(history: list, keep_last_n_turns: int = KEEP_LAST_N_TURNS) -> list:
    """
    history 안에서 role='user' 인 딕셔너리 메시지를 '턴의 시작점'으로 보고,
    최근 keep_last_n_turns 개의 사용자 턴부터의 내용만 남기고 그 이전은 잘라낸다.
    (함수콜 -> 함수결과 쌍이 끊기지 않도록, 항상 user 메시지 위치에서만 자름)
    """
    if not history:
        return history

    user_indices = [
        i for i, item in enumerate(history)
        if isinstance(item, dict) and item.get("role") == "user"
    ]

    if len(user_indices) <= keep_last_n_turns:
        return history

    cutoff = user_indices[-keep_last_n_turns]
    return history[cutoff:]


# ==========================================================
# web_search / file_search를 정말 필요할 때만 옵션에 넣기 위한 키워드 감지
# (프롬프트로 "검색 자제해" 라고만 하면 GPT가 안 지키는 경우가 많아서,
#  아예 필요 없어 보이면 툴 목록에서 빼서 호출 자체를 물리적으로 막는다)
# ==========================================================
_WEB_SEARCH_KEYWORDS = [
    "시세", "요즘", "최근", "동향", "뉴스", "실시간", "요새", "트렌드",
    "지금", "오늘", "이번주", "이번 주", "이번달", "이번 달",
]
_FILE_SEARCH_KEYWORDS = [
    "체크리스트", "가이드", "주의사항", "유의사항", "확인해야", "구매 팁",
    "체크할", "확인할", "고려해야", "고려사항",
]


def _wants_web_search(user_message: str) -> bool:
    return any(k in user_message for k in _WEB_SEARCH_KEYWORDS)


def _wants_file_search(user_message: str) -> bool:
    return any(k in user_message for k in _FILE_SEARCH_KEYWORDS)


def _collect_recent_user_text(trimmed_history: list, user_message: str) -> str:
    """
    [멀티턴 검색 의도 유지] 사용자가 1턴에서 "시세/최근 동향"이라고 물어봤는데
    2~3턴에서 스펙만 추가로 알려주면(예: "16gb, 256gb"), 그 턴만 보면 키워드가 없어서
    web_search가 꺼져버리는 문제가 있었음.
    -> 최근 대화에 남아있는 사용자 발화들 + 이번 메시지를 합쳐서 판단하면,
       한 번 검색 의도가 생기면 그 정보 수집이 끝날 때까지(트리밍 범위 내에서) 유지됨.
    """
    user_texts = [
        item.get("content", "")
        for item in trimmed_history
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str)
    ]
    user_texts.append(user_message)
    return " ".join(user_texts)


# ==========================================================
# Function Calling용 tools 정의 (predict_price / detect_anomaly)
# enum 값들은 data/processed/ebay_laptops_clean_processed.csv 의 실제 카테고리 값 기준
# (brand 33종 / model_family 48종 중 자주 나오는 것 위주로만 enum에 노출 -> 종류가 너무 많아서... 일단은 이렇게 처리를 해뒀는데.. 리팩토링 할 방법을 찾아보겠습니다.
# 그 외 값은 자유 텍스트로 입력받아도 매핑 실패 시 자동으로 0 처리됨)
# ==========================================================
CUSTOM_TOOLS = [
    {
        "type": "function",
        "name": "predict_price",
        "description": "입력받은 중고 노트북 정보를 기반으로 머신러닝 모델을 호출하여 적정 중고가(달러)를 예측한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "brand": {
                    "type": "string",
                    "description": "제조사",
                    "enum": [
                        "dell", "hp", "lenovo", "apple", "asus", "acer", "toshiba",
                        "samsung", "msi", "microsoft", "sony", "panasonic", "gigabyte",
                        "fujitsu", "lg", "razer", "other",
                    ],
                },
                "model_family": {
                    "type": "string",
                    "description": "제품군 (예: thinkpad, macbook_pro, xps, latitude, ideapad, pavilion 등)",
                },
                "cpu_family": {
                    "type": "string",
                    "description": "CPU 등급",
                    "enum": [
                        "core_i3", "core_i5", "core_i7", "core_i9",
                        "ryzen_3", "ryzen_5", "ryzen_7", "ryzen_9",
                        "celeron", "pentium", "atom", "xeon",
                        "apple_m1", "apple_m2", "apple_m3", "other", "unknown",
                    ],
                },
                "ram_gb": {
                    "type": "number",
                    "description": "RAM 용량(GB)",
                },
                "storage_type": {
                    "type": "string",
                    "description": "저장장치 종류",
                    "enum": ["ssd", "nvme_ssd", "hdd", "emmc", "hybrid", "unknown"],
                },
                "storage_capacity_gb": {
                    "type": "number",
                    "description": "저장용량(GB)",
                },
                "screen_size_inch": {
                    "type": "number",
                    "description": "화면 크기(인치)",
                },
                "os": {
                    "type": "string",
                    "description": "운영체제",
                    "enum": ["windows", "macos", "chrome_os", "linux", "android", "other", "unknown"],
                },
                "condition": {
                    "type": "string",
                    "description": "제품 상태 등급",
                    "enum": ["New", "Open Box", "Refurbished", "Used", "For Parts Or Not Working"],
                },
                "release_year": {
                    "type": "integer",
                    "description": (
                        "출시연도. 사용자가 직접 말하지 않았다면 브랜드/제품군/CPU 세대 등 "
                        "네가 이미 알고 있는 지식을 바탕으로 가장 가능성 높은 연도를 스스로 추정해서 채워라. "
                        "정말 추정할 근거가 전혀 없을 때만 -1을 사용한다."
                    ),
                },
            },
            "required": [
                "brand", "model_family", "cpu_family", "ram_gb", "storage_type",
                "storage_capacity_gb", "screen_size_inch", "os", "condition", "release_year",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "detect_anomaly",
        "description": "predict_price()로 얻은 예측 가격(및 모델 오차 정보)과 사용자가 제시한 판매 가격을 비교하여 저가/적정가/고가 여부를 판단한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_price": {
                    "type": "number",
                    "description": "predict_price 함수가 반환한 예측 적정가(달러)",
                },
                "selling_price": {
                    "type": "number",
                    "description": "사용자가 제시한 실제 판매 가격(달러)",
                },
                "residual_std": {
                    "type": "number",
                    "description": "predict_price 함수가 반환한 모델 오차의 표준편차",
                },
            },
            "required": ["predicted_price", "selling_price", "residual_std"],
            "additionalProperties": False,
        },
    },
]

WEB_SEARCH_TOOL = {
    "type": "web_search_preview",
    "search_context_size": "low",
}

FILE_SEARCH_TOOL = {
    "type": "file_search",
    "vector_store_ids": [VECTOR_STORE_ID],
}


# ==========================================================
# 시스템 프롬프트 (매 호출마다 오늘 날짜 + 실시간 환율을 박아서 새로 생성)
# ==========================================================
def build_system_prompt(usd_krw_rate: float, rate_date: str, today_str: str,
                         web_search_enabled: bool, file_search_enabled: bool) -> str:
    tool_status_lines = []
    if web_search_enabled:
        tool_status_lines.append(f"- web_search: 이번 질문에서 딱 {MAX_WEB_SEARCH_PER_QUESTION}번만 쓸 수 있음. 결과 나오면 바로 답변 마무리.")
    else:
        tool_status_lines.append("- web_search: 이번 질문에는 아예 연결 안 되어 있음 (호출 시도해도 실행 안 됨). 실시간 시세가 꼭 필요하면 '최근 시세를 알려주면 더 정확하다' 정도만 답변에 언급하고, 검색 없이 predict_price 결과로 답변을 마무리해.")
    if file_search_enabled:
        tool_status_lines.append(f"- file_search: 이번 질문에서 딱 {MAX_FILE_SEARCH_PER_QUESTION}번만 쓸 수 있음.")
    else:
        tool_status_lines.append("- file_search: 이번 질문에는 연결 안 되어 있음. 체크리스트가 필요하면 네가 아는 일반적인 중고 노트북 구매 체크포인트로 답변해.")

    tool_status_text = "\n".join(tool_status_lines)

    return f"""
너는 중고 노트북 가격 상담 어시스턴트야.

[오늘 날짜 / 환율 정보]
- 오늘 날짜: {today_str}
- 현재 적용할 USD->KRW 환율: 1 USD = {usd_krw_rate:,.2f} KRW (기준일: {rate_date})
- 원화 환산이 필요하면 반드시 위에 제시된 환율 숫자만 사용해서 계산해. 네가 알고 있는 예전 환율이나
  추측값을 쓰지 말고, web_search로 환율을 다시 찾지도 마 (환율은 이미 위에 정확히 제공됨).

[이번 질문에서 사용 가능한 검색 툴 상태]
{tool_status_text}

규칙 (중요도 순):
1. 절대 네 지식만으로 최종 가격을 추측하지 마 (predict_price 호출 없이 답변 금지).
2. 사용자가 이미 명시적으로 말했거나, 제품명/표현 자체에서 명확히 확정되는 값은 절대 다시 묻거나
   "맞나요?" 식으로 재확인하지 마. 확정된 값으로 그냥 써. 예:
   - "맥북에어", "맥북 에어", "macbook air" -> brand=apple, model_family=macbook_air, os=macos 확정
   - "맥북프로" -> brand=apple, model_family=macbook_pro, os=macos 확정
   - "m1/m2/m3/m4" 언급 -> cpu_family=apple_m1/m2/m3/m4 확정
   - "i5/i7/라이젠5" 등 CPU 언급 -> cpu_family 확정
   - "ssd", "hdd", "nvme" 등 언급 -> storage_type 확정
   - 숫자로 준 RAM(예: "16기가") -> ram_gb=16 확정
   - 숫자로 준 저장용량(예: "512기가") -> storage_capacity_gb=512 확정
   - 원화 금액(예: "300만원") -> selling_price로 그대로 사용 (환율은 위 정보로 직접 환산, 되묻지 마)
   재확인 질문은 사용자를 짜증나게 할 뿐이니 절대 하지 마라.
3. 위 방식으로 확정해도 정말 남는 정보가 없는 항목(예: condition, 또는 같은 모델에 화면 크기가
   여러 종류라 진짜 헷갈리는 경우의 screen_size_inch)만 딱 그 항목만 짧게 물어봐. 이미 확정된
   항목을 다시 나열하며 물어보지 마.
3-1. 아주 중요: 이전 턴에서 이미 확정된 스펙(브랜드, 모델군, CPU 등)은 이번 사용자 메시지에
     다시 언급되지 않아도 절대 잊지 말고 그대로 유지해서 predict_price에 넣어. 대화 기록 전체를
     보고 지금까지 나온 모든 확정 정보를 합쳐서 판단해. 예를 들어 이전 턴에 "m3"라고 했는데
     이번 턴에 "16기가, 512기가 ssd"만 왔다면 cpu_family는 여전히 apple_m3여야 하고 절대
     unknown으로 바뀌면 안 돼.
4. 딱 하나, release_year(출시연도)만 예외야. 사용자가 출시연도를 말하지 않았다면 절대 되묻지 말고,
   brand/model_family/cpu_family 등 이미 확보한 정보로 네가 알고 있는 지식을 바탕으로 가장 그럴듯한
   연도를 스스로 추정해서 채워. 정말 추정할 근거가 전혀 없을 때만 -1을 사용해.
5. 필요한 값이 다 확정되면(release_year는 3번 규칙으로 자동 추정되니 기다릴 필요 없음)
   추가 질문 없이 즉시 predict_price를 호출해. 다 모였는데 또 질문으로 넘어가지 마.
6. 판매가(selling_price)까지 알게 되면 반드시 detect_anomaly 함수를 호출해서 저가/적정가/고가를 판정해.
   사용자가 판매가를 말한 적이 전혀 없다면 절대로 0이나 임의의 숫자를 지어내서 detect_anomaly를
   호출하지 마 (predict_price 결과만 안내하고 "판매가를 알려주시면 저가/적정가/고가까지 판정해드릴게요"
   라고만 덧붙여).
7. 함수 호출 결과를 받으면, 그 결과를 근거로 최종 답변(적정가, 판정, 이유)을 제공하고 바로 마무리해.
   - 최종 답변에는 항상 "달러 원본 예측가"와 "위에 제시된 환율로 환산한 원화 금액"을 함께 보여줘.
   - release_year를 네가 추정해서 채운 경우에는 "출시연도는 약 OOOO년으로 추정해서 계산했다" 정도로 짧게 언급해.
8. web_search/file_search는 위 [사용 가능한 검색 툴 상태]에 나온 범위 안에서만 사용해. 그 외에는 절대 호출하지 마.
"""


# 사용자 메시지를 받아서
# 에이전트에게 tools(predict_price, detect_anomaly, web_search, file_search)와 함께 전달
# 에이전트가 커스텀 함수(predict_price, detect_anomaly) 호출을 요청하면 우리가 직접 실행하고 결과를 다시 에이전트에게 돌려줌
# (호스팅 툴 web_search/file_search는 OpenAI가 알아서 처리하므로 신경 안 써도 됨)
# 에이전트가 더 이상 함수 호출이 필요없으면 자연어로 출력해서 사용자에게 반환
def orchestrate(user_message: str, conversation_history: list = None) -> tuple:
    if conversation_history is None:
        conversation_history = []

    # [속도 개선] 예전 턴을 계속 누적해서 보내면 매번 느려지므로, 최근 N턴만 남기고 자름
    trimmed_history = trim_conversation_history(conversation_history, KEEP_LAST_N_TURNS)

    # [환율] 캐싱된 실시간 환율 조회 (6시간 지나야 다시 API 호출)
    usd_krw_rate, rate_date = get_usd_krw_rate()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # [web_search/file_search 남용 차단 1단계] 사용자 질문에 관련 키워드가 없으면
    # 아예 이번 질문에서 옵션 자체를 안 줌 (GPT가 호출하고 싶어도 물리적으로 불가능)
    # 이번 메시지 하나만 보면 멀티턴 중간에 검색 의도가 꺼져버리므로, 최근 대화 전체를 합쳐서 판단.
    combined_user_text = _collect_recent_user_text(trimmed_history, user_message)
    web_search_available = _wants_web_search(combined_user_text)
    file_search_available = _wants_file_search(combined_user_text)

    # 이번 질문(orchestrate 호출) 동안 실제로 검색을 몇 번 썼는지 카운트 (2단계 차단용)
    web_search_used_count = 0
    file_search_used_count = 0

    # history에는 developer 프롬프트를 저장하지 않고, 매 호출마다 새로 맨 앞에 붙임
    input_messages = trimmed_history + [{"role": "user", "content": user_message}]

    round_num = 0

    while True:
        round_num += 1

        # [web_search/file_search 남용 차단 2단계] 이번 라운드에 실제로 넣어줄 툴 목록을 매번 다시 계산.
        # 이미 한도만큼 썼으면 다음 라운드부터는 목록에서 완전히 빼버림 -> 재검색 자체가 불가능해짐.
        current_tools = list(CUSTOM_TOOLS)
        if web_search_available and web_search_used_count < MAX_WEB_SEARCH_PER_QUESTION:
            current_tools.append(WEB_SEARCH_TOOL)
        if file_search_available and file_search_used_count < MAX_FILE_SEARCH_PER_QUESTION:
            current_tools.append(FILE_SEARCH_TOOL)

        system_prompt = build_system_prompt(
            usd_krw_rate, rate_date, today_str,
            web_search_enabled=(web_search_available and web_search_used_count < MAX_WEB_SEARCH_PER_QUESTION),
            file_search_enabled=(file_search_available and file_search_used_count < MAX_FILE_SEARCH_PER_QUESTION),
        )

        print(f"\n[디버그] ===== {round_num}번째 API 호출 시작 =====")
        print(f"[디버그] 이번에 API로 보내는 메시지 개수: {len(input_messages) + 1} (developer 프롬프트 포함)")
        print(f"[디버그] 이번 라운드 web_search 사용 가능: {web_search_available and web_search_used_count < MAX_WEB_SEARCH_PER_QUESTION} / file_search 사용 가능: {file_search_available and file_search_used_count < MAX_FILE_SEARCH_PER_QUESTION}")

        response = client.responses.create(
            model="gpt-5",
            tools=current_tools,
            input=[{"role": "developer", "content": system_prompt}] + input_messages,
            # [속도 개선] gpt-5 기본 reasoning은 느림. 이 작업은 규칙대로 함수만 잘 호출하면 되는 수준이라
            # 깊은 추론이 필요 없음. 단, reasoning.effort="minimal"은 file_search/web_search와 함께 못 쓰는
            # OpenAI 제약이 있어서, 검색 툴이 이번 라운드에 포함되는지 여부에 따라 effort를 다르게 준다.
            reasoning={"effort": "minimal" if len(current_tools) == len(CUSTOM_TOOLS) else "low"},
            text={"verbosity": "low"},
        )

        output_types = [item.type for item in response.output]
        print(f"[디버그] 이번 응답에 포함된 항목 타입들: {output_types}")

        # 이번 라운드에서 실제로 웹서치/파일서치를 썼는지 카운트 갱신 (2단계 차단의 핵심)
        web_search_used_count += sum(1 for t in output_types if t == "web_search_call")
        file_search_used_count += sum(1 for t in output_types if t == "file_search_call")

        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            print(f"[디버그] -> 함수 호출 요청 없음. 최종 텍스트 답변 생성함.")
            print(f"[디버그] -> 이번 응답의 output_text 앞부분: {response.output_text[:80]}...")
            input_messages += response.output
            return response.output_text, input_messages

        print(f"[디버그] -> 이번 응답에서 호출 요청된 함수: {[call.name for call in function_calls]}")

        input_messages += response.output

        for call in function_calls:
            func = AVAILABLE_FUNCTIONS.get(call.name)
            args = json.loads(call.arguments)
            print(f"[디버그]    실행 중: {call.name}({args})")

            result = func(**args) if func else {"error": f"알 수 없는 함수: {call.name}"}
            print(f"[디버그]    실행 결과: {result}")

            input_messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })

        print(f"[디버그] -> 함수 실행 결과를 input_messages에 추가함. 다음 반복에서 이 결과를 GPT에게 다시 보여줌.")


# ==========================================================
# 메인 실행부
# ==========================================================
if __name__ == "__main__":
    print("중고 노트북 가격 어시스턴트입니다. 종료하려면 'exit' 입력하세요.\n")
    history = []

    while True:
        user_message = input("질문을 입력하세요 (예: 델 래티튜드 i5 16GB 512GB SSD Used 50만원이면 괜찮아?): ")

        if user_message.strip().lower() in ("exit", "quit", "종료"):
            break

        answer, history = orchestrate(user_message, history)

        print("==================== 답변 ====================")
        print(answer)