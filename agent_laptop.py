"""
agent.py

<필요 함수>
predict_price_laptop(), detect_anomaly(): predict.py 에서 import만 하기
web_search / file_search: 호스팅 툴로 등록
orchestrate(): 대화 오케스트레이션 (Function Calling 왕복 처리)
"""

# ==========================================================
# 모듈 import (전부 파일 최상단에 모음)
# ==========================================================
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from predict_laptop import predict_price_laptop, detect_anomaly   # predict_laptop.py에서 가져오기

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
    "predict_price_laptop": predict_price_laptop,
    "detect_anomaly": detect_anomaly
}

# ==========================================================
# Function Calling용 tools 정의
# enum 값들은 data/processed/ebay_laptops_clean_processed.csv 의 실제 카테고리 값 기준
# (brand 33종 / model_family 48종 중 자주 나오는 것 위주로만 enum에 노출 -> 종류가 너무 많아서... 일단은 이렇게 처리를 해뒀는데.. 리팩토링 할 방법을 찾아보겠습니다.
# 그 외 값은 자유 텍스트로 입력받아도 매핑 실패 시 자동으로 0 처리됨)
# ==========================================================
tools = [
    {
        "type": "function",
        "name": "predict_price_laptop",
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
                    "description": "출시연도. 모르면 -1",
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
        "description": "predict_price_laptop()로 얻은 예측 가격(및 모델 오차 정보)과 사용자가 제시한 판매 가격을 비교하여 저가/적정가/고가 여부를 판단한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_price": {
                    "type": "number",
                    "description": "predict_price_laptop 함수가 반환한 예측 적정가(달러)",
                },
                "selling_price": {
                    "type": "number",
                    "description": "사용자가 제시한 실제 판매 가격(달러)",
                },
                "residual_std": {
                    "type": "number",
                    "description": "predict_price_laptop 함수가 반환한 모델 오차의 표준편차",
                },
            },
            "required": ["predicted_price", "selling_price", "residual_std"],
            "additionalProperties": False,
        },
    },
]

# ==========================================================
# 시스템 프롬프트 (에이전트 동작 규칙)
# ==========================================================
SYSTEM_PROMPT = """
너는 중고 노트북 가격 상담 어시스턴트야.
 
규칙:
1. 절대 네 지식만으로 가격을 추측하지 마.
2. 브랜드, 제품군, CPU, RAM, 저장장치, 화면크기, OS, 상태를 알게 되면 반드시 predict_price_laptop 함수를 호출해.
3. 판매가(selling_price)까지 알게 되면 반드시 detect_anomaly 함수를 호출해서 저가/적정가/고가를 판정해.
4. 이미 필수 정보가 사용자 메시지에 다 있으면, 추가 질문 없이 바로 함수부터 호출해.
5. 정말 필수 정보가 빠졌을 때만 되물어.
6. 함수 호출 결과를 받으면, 그 결과를 근거로 최종 답변(적정가, 판정, 이유)을 제공하고 대화를 마무리해.
7. 최신 시세 동향이나 실시간 정보가 필요하면 web_search를 사용하되, 최대 1~2번만 검색하고 그 결과로 답변해. 여러 번 반복 검색하지 마.
8. 구매 시 확인해야 할 체크리스트나 가이드가 필요하면 file_search를 사용해.
9. 가격 단위는 달러(USD)로 학습된 모델이니, 사용자가 원화로 물어보면 예측 결과를 대략적인 환율로 환산해서 안내해.
10. 사용자가 시세 동향만 물어봐도, 스펙(브랜드/CPU/RAM/저장장치 등)을 알 수 있다면
    predict_price_laptop도 함께 호출해서 우리 모델의 예측치와 웹 검색 결과를 같이 제시해.
    이러면 사용자가 두 가지 관점(실제 매물 시세 vs 모델 예측 적정가)을 다 볼 수 있어.
"""

# 사용자 메시지를 받아서
# 에이전트에게 tools(predict_price_laptop, detect_anomaly, web_search, file_search)와 함께 전달
# 에이전트가 커스텀 함수(predict_price_laptop, detect_anomaly) 호출을 요청하면 우리가 직접 실행하고 결과를 다시 에이전트에게 돌려줌
# (호스팅 툴 web_search/file_search는 OpenAI가 알아서 처리하므로 신경 안 써도 됨)
# 에이전트가 더 이상 함수 호출이 필요없으면 자연어로 출력해서 사용자에게 반환
# refactor: 이전까지의 대화 기록 누적
def orchestrate(user_message: str, conversation_history: list) -> tuple:
    if conversation_history is None:
        conversation_history = []

    # 커스텀 툴이랑 호스팅 툴 전달
    all_tools = tools + [
        {
            "type": "web_search_preview",
            # 검색 컨텍스트를 줄여서 검색 횟수/깊이 제한
            "search_context_size": "low",
        },
        {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]},
    ]

    # 대화가 처음 시작될 때만 시스템 프롬프트를 맨 앞에 넣음
    if not conversation_history:
        # 대화 기록 (에이전트 응답, 함수 실행 결과가 계속 쌓임)
        input_messages = conversation_history + [
            # 이게 없으면 GPT가 함수 안 쓰고 알아서 답함
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    else:
        input_messages = conversation_history + [{"role": "user", "content": user_message}]

    # 반복 횟수 확인용
    round_num = 0

    # 에이전트가 함수 호출을 계속 요청할 경우 -> 무한 루프로 처리
    while True:
        round_num += 1
        print(f"\n[디버그] ===== {round_num}번째 API 호출 시작 =====")
        print(f"[디버그] 지금까지 쌓인 input_messages 개수: {len(input_messages)}")

        response = client.responses.create(
            model="gpt-5.5",          # 모델 수정 5 -> 5.5
            tools=all_tools,
            input=input_messages,
        )

        # response.output에 있는 모든 항목의 타입을 확인
        print(f"[디버그] 이번 응답에 포함된 항목 타입들: {[item.type for item in response.output]}")

        # 웅답에 커스텀 함수 호출 부탁이 있었는지
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            print(f"[디버그] -> 함수 호출 요청 없음. GPT가 자체 지식/이전 함수 결과로 최종 텍스트 답변 생성함.")
            print(f"[디버그] -> 이번 응답의 output_text 앞부분: {response.output_text[:80]}...")
            # 최종 답변도 기록에 추가
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