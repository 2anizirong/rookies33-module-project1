# """
# agent.py

# <필요 함수>
# predict_price(), detect_anomaly(): predict.py 에서 import만 하기
# web_search / file_search: 호스팅 툴로 등록
# orchestrate(): 대화 오케스트레이션 (Function Calling 왕복 처리)
# """

# # ==========================================================
# # 모듈 import (전부 파일 최상단에 모음)
# # ==========================================================
# import json
# import os

# from dotenv import load_dotenv
# from openai import OpenAI

# from predict import predict_price, detect_anomaly   # predict.py에서 가져오기

# # ==========================================================
# # 상수 / 클라이언트
# # ==========================================================

# # .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
# load_dotenv()

# # OpenAI API Client 생성 - 환경변수에서 키를 불러와 사용 (하드코딩 금지)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

# # File Search에서 사용할 Vector Store ID (이것도 .env에서 관리하는 걸 추천)
# VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# # 실제 함수 이름으로 함수 객체 매핑하기 (agent가 호출 요청하면 여기서 실행)
# AVAILABLE_FUNCTIONS = {
#     "predict_price": predict_price,
#     "detect_anomaly": detect_anomaly
# }

# # ==========================================================
# # Function Calling용 tools 정의
# # enum 값들은 data/processed/ebay_laptops_clean_processed.csv 의 실제 카테고리 값 기준
# # (brand 33종 / model_family 48종 중 자주 나오는 것 위주로만 enum에 노출 -> 종류가 너무 많아서... 일단은 이렇게 처리를 해뒀는데.. 리팩토링 할 방법을 찾아보겠습니다.
# # 그 외 값은 자유 텍스트로 입력받아도 매핑 실패 시 자동으로 0 처리됨)
# # ==========================================================
# tools = [
#     {
#         "type": "function",
#         "name": "predict_price",
#         "description": "입력받은 중고 노트북 정보를 기반으로 머신러닝 모델을 호출하여 적정 중고가(달러)를 예측한다.",
#         "strict": True,
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "brand": {
#                     "type": "string",
#                     "description": "제조사",
#                     "enum": [
#                         "dell", "hp", "lenovo", "apple", "asus", "acer", "toshiba",
#                         "samsung", "msi", "microsoft", "sony", "panasonic", "gigabyte",
#                         "fujitsu", "lg", "razer", "other",
#                     ],
#                 },
#                 "model_family": {
#                     "type": "string",
#                     "description": "제품군 (예: thinkpad, macbook_pro, xps, latitude, ideapad, pavilion 등)",
#                 },
#                 "cpu_family": {
#                     "type": "string",
#                     "description": "CPU 등급",
#                     "enum": [
#                         "core_i3", "core_i5", "core_i7", "core_i9",
#                         "ryzen_3", "ryzen_5", "ryzen_7", "ryzen_9",
#                         "celeron", "pentium", "atom", "xeon",
#                         "apple_m1", "apple_m2", "apple_m3", "other", "unknown",
#                     ],
#                 },
#                 "ram_gb": {
#                     "type": "number",
#                     "description": "RAM 용량(GB)",
#                 },
#                 "storage_type": {
#                     "type": "string",
#                     "description": "저장장치 종류",
#                     "enum": ["ssd", "nvme_ssd", "hdd", "emmc", "hybrid", "unknown"],
#                 },
#                 "storage_capacity_gb": {
#                     "type": "number",
#                     "description": "저장용량(GB)",
#                 },
#                 "screen_size_inch": {
#                     "type": "number",
#                     "description": "화면 크기(인치)",
#                 },
#                 "os": {
#                     "type": "string",
#                     "description": "운영체제",
#                     "enum": ["windows", "macos", "chrome_os", "linux", "android", "other", "unknown"],
#                 },
#                 "condition": {
#                     "type": "string",
#                     "description": "제품 상태 등급",
#                     "enum": ["New", "Open Box", "Refurbished", "Used", "For Parts Or Not Working"],
#                 },
#                 "release_year": {
#                     "type": "integer",
#                     "description": "출시연도. 모르면 -1",
#                 },
#             },
#             "required": [
#                 "brand", "model_family", "cpu_family", "ram_gb", "storage_type",
#                 "storage_capacity_gb", "screen_size_inch", "os", "condition", "release_year",
#             ],
#             "additionalProperties": False,
#         },
#     },
#     {
#         "type": "function",
#         "name": "detect_anomaly",
#         "description": "predict_price()로 얻은 예측 가격(및 모델 오차 정보)과 사용자가 제시한 판매 가격을 비교하여 저가/적정가/고가 여부를 판단한다.",
#         "strict": True,
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "predicted_price": {
#                     "type": "number",
#                     "description": "predict_price 함수가 반환한 예측 적정가(달러)",
#                 },
#                 "selling_price": {
#                     "type": "number",
#                     "description": "사용자가 제시한 실제 판매 가격(달러)",
#                 },
#                 "residual_std": {
#                     "type": "number",
#                     "description": "predict_price 함수가 반환한 모델 오차의 표준편차",
#                 },
#             },
#             "required": ["predicted_price", "selling_price", "residual_std"],
#             "additionalProperties": False,
#         },
#     },
# ]

# # ==========================================================
# # 시스템 프롬프트 (에이전트 동작 규칙)
# # ==========================================================
# SYSTEM_PROMPT = """
# 너는 중고 노트북 가격 상담 어시스턴트야.
 
# 규칙:
# 1. 절대 네 지식만으로 가격을 추측하지 마.
# 2. 브랜드, 제품군, CPU, RAM, 저장장치, 화면크기, OS, 상태를 알게 되면 반드시 predict_price 함수를 호출해.
# 3. 판매가(selling_price)까지 알게 되면 반드시 detect_anomaly 함수를 호출해서 저가/적정가/고가를 판정해.
# 4. 이미 필수 정보가 사용자 메시지에 다 있으면, 추가 질문 없이 바로 함수부터 호출해.
# 5. 정말 필수 정보가 빠졌을 때만 되물어.
# 6. 함수 호출 결과를 받으면, 그 결과를 근거로 최종 답변(적정가, 판정, 이유)을 제공하고 대화를 마무리해.
# 7. 최신 시세 동향이나 실시간 정보가 필요하면 web_search를 사용하되, 최대 1~2번만 검색하고 그 결과로 답변해. 여러 번 반복 검색하지 마.
# 8. 구매 시 확인해야 할 체크리스트나 가이드가 필요하면 file_search를 사용해.
# 9. 가격 단위는 달러(USD)로 학습된 모델이니, 사용자가 원화로 물어보면 예측 결과를 대략적인 환율로 환산해서 안내해.
# 10. 사용자가 시세 동향만 물어봐도, 스펙(브랜드/CPU/RAM/저장장치 등)을 알 수 있다면
#     predict_price도 함께 호출해서 우리 모델의 예측치와 웹 검색 결과를 같이 제시해.
#     이러면 사용자가 두 가지 관점(실제 매물 시세 vs 모델 예측 적정가)을 다 볼 수 있어.
# """

# # 사용자 메시지를 받아서
# # 에이전트에게 tools(predict_price, detect_anomaly, web_search, file_search)와 함께 전달
# # 에이전트가 커스텀 함수(predict_price, detect_anomaly) 호출을 요청하면 우리가 직접 실행하고 결과를 다시 에이전트에게 돌려줌
# # (호스팅 툴 web_search/file_search는 OpenAI가 알아서 처리하므로 신경 안 써도 됨)
# # 에이전트가 더 이상 함수 호출이 필요없으면 자연어로 출력해서 사용자에게 반환
# # refactor: 이전까지의 대화 기록 누적
# def orchestrate(user_message: str, conversation_history: list) -> tuple:
#     if conversation_history is None:
#         conversation_history = []

#     # 커스텀 툴이랑 호스팅 툴 전달
#     all_tools = tools + [
#         {
#             "type": "web_search_preview",
#             # 검색 컨텍스트를 줄여서 검색 횟수/깊이 제한
#             "search_context_size": "low",
#         },
#         {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]},
#     ]

#     # 대화가 처음 시작될 때만 시스템 프롬프트를 맨 앞에 넣음
#     if not conversation_history:
#         # 대화 기록 (에이전트 응답, 함수 실행 결과가 계속 쌓임)
#         input_messages = conversation_history + [
#             # 이게 없으면 GPT가 함수 안 쓰고 알아서 답함
#             {"role": "developer", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_message}
#         ]
#     else:
#         input_messages = conversation_history + [{"role": "user", "content": user_message}]

#     # 반복 횟수 확인용
#     round_num = 0

#     # 에이전트가 함수 호출을 계속 요청할 경우 -> 무한 루프로 처리
#     while True:
#         round_num += 1
#         print(f"\n[디버그] ===== {round_num}번째 API 호출 시작 =====")
#         print(f"[디버그] 지금까지 쌓인 input_messages 개수: {len(input_messages)}")

#         response = client.responses.create(
#             model="gpt-5",          # 모델은 위에 쓴 모델이랑 통일
#             tools=all_tools,
#             input=input_messages,
#         )

#         # response.output에 있는 모든 항목의 타입을 확인
#         print(f"[디버그] 이번 응답에 포함된 항목 타입들: {[item.type for item in response.output]}")

#         # 웅답에 커스텀 함수 호출 부탁이 있었는지
#         function_calls = [item for item in response.output if item.type == "function_call"]

#         if not function_calls:
#             print(f"[디버그] -> 함수 호출 요청 없음. GPT가 자체 지식/이전 함수 결과로 최종 텍스트 답변 생성함.")
#             print(f"[디버그] -> 이번 응답의 output_text 앞부분: {response.output_text[:80]}...")
#             # 최종 답변도 기록에 추가
#             input_messages += response.output13.6
#             return response.output_text, input_messages

#         print(f"[디버그] -> 이번 응답에서 호출 요청된 함수: {[call.name for call in function_calls]}")

#         # 이전 응답에 대화 기록 추가
#         input_messages += response.output

#         # 이제 하나씩 실행
#         for call in function_calls:
#             func = AVAILABLE_FUNCTIONS.get(call.name)
#             args = json.loads(call.arguments)
#             print(f"[디버그]    실행 중: {call.name}({args})")

#             result = func(**args) if func else {"error": f"알 수 없는 함수: {call.name}"}
#             print(f"[디버그]    실행 결과: {result}")

#             # 함수 실행 결과를 대화 기록에 추가 (에이전트가 다음 턴에 이 결과를 보고 답변 생성)
#             input_messages.append({
#                 "type": "function_call_output",
#                 "call_id": call.call_id,
#                 "output": json.dumps(result, ensure_ascii=False),
#             })

#         print(f"[디버그] -> 함수 실행 결과를 input_messages에 추가함. 다음 반복에서 이 결과를 GPT에게 다시 보여줌.")

# # ==========================================================
# # 메인 실행부
# # ==========================================================
# if __name__ == "__main__":
#     print("중고 노트북 가격 어시스턴트입니다. 종료하려면 'exit' 입력하세요.\n")
#     history = []
 
#     while True:
#         user_message = input("질문을 입력하세요 (예: 델 래티튜드 i5 16GB 512GB SSD Used 50만원이면 괜찮아?): ")
 
#         if user_message.strip().lower() in ("exit", "quit", "종료"):
#             break
 
#         answer, history = orchestrate(user_message, history)
 
#         print("==================== 답변 ====================")
#         print(answer)


"""
agent.py

<필요 함수>
predict_price(), detect_anomaly(): predict.py 에서 import만 하기
web_search / file_search: 호스팅 툴로 등록
orchestrate(): 대화 오케스트레이션 (Function Calling 왕복 처리)

<이번 수정 사항>
1. [속도 개선] 대화가 길어질수록 매 API 호출마다 input_messages 전체(누적된 함수콜/웹서치 결과까지)를
   그대로 다시 보내고 있어서 턴이 쌓일수록 토큰 수가 계속 늘어나 느려졌음.
   -> trim_conversation_history()로 "최근 N번의 사용자 턴"만 남기고 예전 기록은 잘라냄.
   -> developer(system) 프롬프트도 매 호출마다 새로 앞에 붙이고, 저장되는 history에는 포함 안 시켜서
      history 자체가 불필요하게 커지는 것도 방지.
2. [환율] 원래는 시스템 프롬프트에 "대략 환율로 환산해" 라고만 되어 있어서 GPT가 부정확한 값으로
   추측하거나 web_search를 써서 왕복이 늘어남 -> get_usd_krw_rate()로 실제 환율 API를 호출해
   "오늘 날짜 기준 실제 환율"을 가져오고, 이 값을 시스템 프롬프트에 직접 박아서 GPT가 그 숫자로만
   계산하도록 함 (6시간 캐싱해서 매번 API 호출하지 않도록 처리 -> 이것도 속도에 도움).
3. [출시연도] 사용자가 출시연도를 모를 수 있으니, GPT가 브랜드/제품군/CPU 정보를 바탕으로 스스로
   추정하도록 시스템 프롬프트 규칙과 tools의 파라미터 설명을 수정함 (더 이상 무조건 -1이 아님).
4. [버그 수정] `input_messages += response.output13.6` 오타로 되어 있던 부분 -> `response.output`으로 수정
   (이 오타 때문에 함수 호출이 필요없는 최종 답변 케이스에서 AttributeError로 죽었을 것)

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

from predict import predict_price, detect_anomaly   # predict.py에서 가져오기

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
# Function Calling용 tools 정의
# enum 값들은 data/processed/ebay_laptops_clean_processed.csv 의 실제 카테고리 값 기준
# (brand 33종 / model_family 48종 중 자주 나오는 것 위주로만 enum에 노출 -> 종류가 너무 많아서... 일단은 이렇게 처리를 해뒀는데.. 리팩토링 할 방법을 찾아보겠습니다.
# 그 외 값은 자유 텍스트로 입력받아도 매핑 실패 시 자동으로 0 처리됨)
# ==========================================================
tools = [
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


# ==========================================================
# 시스템 프롬프트 (매 호출마다 오늘 날짜 + 실시간 환율을 박아서 새로 생성)
# ==========================================================
def build_system_prompt(usd_krw_rate: float, rate_date: str, today_str: str) -> str:
    return f"""
너는 중고 노트북 가격 상담 어시스턴트야.

[오늘 날짜 / 환율 정보]
- 오늘 날짜: {today_str}
- 현재 적용할 USD->KRW 환율: 1 USD = {usd_krw_rate:,.2f} KRW (기준일: {rate_date})
- 원화 환산이 필요하면 반드시 위에 제시된 환율 숫자만 사용해서 계산해. 네가 알고 있는 예전 환율이나
  추측값을 쓰지 말고, web_search로 환율을 다시 찾지도 마 (환율은 이미 위에 정확히 제공됨).

규칙:
1. 절대 네 지식만으로 가격을 추측하지 마.
2. 브랜드, 제품군, CPU, RAM, 저장장치, 화면크기, OS, 상태를 알게 되면 반드시 predict_price 함수를 호출해.
3. 판매가(selling_price)까지 알게 되면 반드시 detect_anomaly 함수를 호출해서 저가/적정가/고가를 판정해.
4. 이미 필수 정보가 사용자 메시지에 다 있으면, 추가 질문 없이 바로 함수부터 호출해.
5. 정말 필수 정보가 빠졌을 때만 되물어.
6. 출시연도(release_year)는 사용자가 모를 수 있어. 사용자가 언급하지 않았다면 절대 되묻지 말고,
   브랜드/제품군/CPU 세대 등 네가 알고 있는 지식으로 가장 그럴듯한 연도를 스스로 추정해서 predict_price에 넣어.
7. 함수 호출 결과를 받으면, 그 결과를 근거로 최종 답변(적정가, 판정, 이유)을 제공하고 대화를 마무리해.
   - 최종 답변에는 항상 "달러 원본 예측가"와 "위에 제시된 환율로 환산한 원화 금액"을 함께 보여줘.
8. 최신 시세 동향이나 실시간 정보(환율 제외)가 필요하면 web_search를 사용하되, 최대 1~2번만 검색하고 그 결과로 답변해. 여러 번 반복 검색하지 마.
9. 구매 시 확인해야 할 체크리스트나 가이드가 필요하면 file_search를 사용해.
10. 사용자가 시세 동향만 물어봐도, 스펙(브랜드/CPU/RAM/저장장치 등)을 알 수 있다면
    predict_price도 함께 호출해서 우리 모델의 예측치와 웹 검색 결과를 같이 제시해.
    이러면 사용자가 두 가지 관점(실제 매물 시세 vs 모델 예측 적정가)을 다 볼 수 있어.
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
    system_prompt = build_system_prompt(usd_krw_rate, rate_date, today_str)

    # 커스텀 툴이랑 호스팅 툴 전달
    all_tools = tools + [
        {
            "type": "web_search_preview",
            # 검색 컨텍스트를 줄여서 검색 횟수/깊이 제한
            "search_context_size": "low",
        },
        {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]},
    ]

    # history에는 developer 프롬프트를 저장하지 않고, 매 호출마다 새로 맨 앞에 붙임
    # (환율/날짜가 매번 최신 값으로 갱신되고, history 자체도 불필요하게 커지지 않음)
    input_messages = trimmed_history + [{"role": "user", "content": user_message}]

    # 반복 횟수 확인용
    round_num = 0

    # 에이전트가 함수 호출을 계속 요청할 경우 -> 무한 루프로 처리
    while True:
        round_num += 1
        print(f"\n[디버그] ===== {round_num}번째 API 호출 시작 =====")
        print(f"[디버그] 이번에 API로 보내는 메시지 개수: {len(input_messages) + 1} (developer 프롬프트 포함)")

        response = client.responses.create(
            model="gpt-5",          # 모델은 위에 쓴 모델이랑 통일
            tools=all_tools,
            input=[{"role": "developer", "content": system_prompt}] + input_messages,
            # [속도 개선] gpt-5는 기본적으로 내부적으로 깊게 추론(reasoning)하느라 느림.
            # 이 에이전트는 정해진 규칙대로 함수만 잘 호출하면 되는 작업이라 깊은 추론이 필요 없으므로
            # effort를 낮추고 답변 장황함(verbosity)도 낮춰서 매 왕복 속도를 줄임.
            # 주의: reasoning.effort="minimal"은 file_search/web_search 같은 호스팅 툴과 함께 못 쓰기 때문에
            # (OpenAI 쪽 제약) 그보다 한 단계 위인 "low"를 사용함.
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )

        # response.output에 있는 모든 항목의 타입을 확인
        print(f"[디버그] 이번 응답에 포함된 항목 타입들: {[item.type for item in response.output]}")

        # 응답에 커스텀 함수 호출 부탁이 있었는지
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            print(f"[디버그] -> 함수 호출 요청 없음. GPT가 자체 지식/이전 함수 결과로 최종 텍스트 답변 생성함.")
            print(f"[디버그] -> 이번 응답의 output_text 앞부분: {response.output_text[:80]}...")
            # 최종 답변도 기록에 추가 (오타 수정: response.output13.6 -> response.output)
            input_messages += response.output
            return response.output_text, input_messages

        print(f"[디버그] -> 이번 응답에서 호출 요청된 함수: {[call.name for call in function_calls]}")

        # 이전 응답에 대화 기록 추가
        input_messages += response.output

        # 이제 하나씩 실행
        for call in function_calls:
            func = AVAILABLE_FUNCTIONS.get(call.name)
            args = json.loads(call.arguments)
            print(f"[디버그]    실행 중: {call.name}({args})")

            result = func(**args) if func else {"error": f"알 수 없는 함수: {call.name}"}
            print(f"[디버그]    실행 결과: {result}")

            # 함수 실행 결과를 대화 기록에 추가 (에이전트가 다음 턴에 이 결과를 보고 답변 생성)
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