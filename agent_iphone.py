"""
agent.py

중고 아이폰 가격 상담 에이전트 (OpenAI Responses API 기반)

<필요 함수>
predict_price_iphone(), detect_anomaly(): predict.py 에서 import만 하기
search_market_price() - web search
search_buying_guide() - file search
generate_result()
"""

# ==========================================================
# 모듈 import (전부 파일 최상단에 모음)
# ==========================================================
import os

from dotenv import load_dotenv       # .env 파일에서 환경변수를 읽어오기 위한 라이브러리
from openai import OpenAI            # OpenAI 공식 SDK

from predict_iphone import predict_price_iphone, detect_anomaly   # predict_iphone.py에서 가져오기 (ML 모델 함수 2개)
import json                          # 함수 호출 인자(JSON 문자열) 파싱 / 결과 직렬화에 사용

# ==========================================================
# 상수 / 클라이언트 / 함수 호출
# ==========================================================

# .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
load_dotenv()

# OpenAI API Client 생성 - 환경변수에서 키를 불러와 사용 (하드코딩 금지)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# File Search에서 사용할 Vector Store ID (이것도 .env에서 관리하는 걸 추천)
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# 실제 함수 이름(문자열) -> 함수 객체 매핑
# 에이전트가 이 함수를 호출해줘라고 요청하면, 여기서 이름으로 실제 파이썬 함수를 찾아 실행함
AVAILABLE_FUNCTIONS = {
    "predict_price_iphone": predict_price_iphone,
    "detect_anomaly": detect_anomaly
}

# ==========================================================
# Function Calling용 tools 정의
# ==========================================================
# OpenAI Responses API에 전달할 커스텀 함수 도구 스펙
tools = [
    {
        "type": "function",
        "name": "predict_price_iphone",
        "description": "입력받은 아이폰 정보를 기반으로 머신러닝 모델을 호출하여 적정 중고가를 예측한다.",
        "strict": True,            # 파라미터 스키마를 엄격하게 검증 (타입/필수값 등)
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "제품명 (예: iPhone 16 Pro)"
                },
                "storage_gb": {
                    "type": "integer",
                    "description": "저장용량(GB)",
                    "enum": [64, 128, 256, 512, 1024, 2048]   # 학습 데이터의 카테고리 값과 반드시 일치해야 함
                },
                "condition": {
                    "type": "string",
                    "description": "제품 상태 등급",
                    "enum": [
                        "New",
                        "Open Box",
                        "Used",
                        "Good - Refurbished",
                        "Very Good - Refurbished",
                        "Excellent - Refurbished", 
                        "For Parts Or Not Working"
                    ]
                }
            },
            "required": ["title", "storage_gb", "condition"],   # 이 3개가 모두 있어야 함수 호출이 가능
            "additionalProperties": False                       # 정의되지 않은 필드는 허용하지 않음
        }
    },
    {
        "type": "function",
        "name": "detect_anomaly",
        "description": "predict_price_iphone()로 얻은 예측 가격(및 모델 오차 정보)과 사용자가 제시한 판매 가격을 비교하여 저가/적정가/고가 여부를 판단한다.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_price": {
                    "type": "number",
                    "description": "predict_price_iphone 함수가 반환한 예측 적정가(원)"
                },
                "selling_price": {
                    "type": "number",
                    "description": "사용자가 제시한 실제 판매 가격(원)"
                },
                "residual_std": {
                    "type": "number",
                    "description": "predict_price_iphone 함수가 반환한 모델 오차의 표준편차"
                }
            },
            "required": ["predicted_price", "selling_price", "residual_std"],
            "additionalProperties": False
        }
    }          
]

# ==========================================================
# 시스템 프롬프트 (에이전트 동작 규칙)
# ==========================================================
# developer 역할 메시지로 전달되어, 에이전트가 지켜야 할 행동 규칙을 명시함
# 이게 없으면 GPT가 함수 호출 없이 자기 지식만으로 답변해버릴 수 있음
SYSTEM_PROMPT = """
너는 중고 아이폰 가격 상담 어시스턴트야.

규칙:
1. 절대 네 지식만으로 가격을 추측하지 마.
2. 제품명(title), 저장용량(storage_gb), 상태(condition)를 알게 되면 반드시 predict_price_iphone 함수를 호출해.
3. 판매가(selling_price)까지 알게 되면 반드시 detect_anomaly 함수를 호출해서 저가/적정가/고가를 판정해.
4. 이미 필수 정보(title, storage_gb, condition, selling_price)가 메시지에 다 있으면, 추가 질문 없이 바로 함수부터 호출해.
5. 정말 필수 정보가 빠졌을 때만 되물어.
6. 함수 호출 결과를 받으면, 그 결과를 근거로 최종 답변(적정가, 판정, 이유)을 제공하고 대화를 마무리해.
7. 웹서치가 필요하면 최대 1~2번만 검색하고, 그 결과로 충분히 답변해. 여러 번 반복 검색하지 마.
8. 최신 시세 동향이나 실시간 정보가 필요하면 web_search를 사용해.
9. 구매 시 확인해야 할 체크리스트나 가이드가 필요하면 file_search를 사용해.
10. 아이폰 거래 관련 문의에만 답변해줘. 다른 건 해주지마.
"""

# 사용자 메시지를 받아서
# 에이전트한테 툴이랑 같이 해서 전달
# 에이전트가 커스텀툴 (2개_predict_price_iphone랑 detect_anomaly) 호출 요청하면 실행해서 결과 에이전트에게 다시 돌려주기
# 에이전트가 더 이상 함수 호출이 필요없으면 자연어로 출력해서 사용자에게 반환
# refactor: 이전까지의 대화 기록 누적
def orchestrate(user_message: str, conversation_history: list) -> tuple:
    """
    사용자 메시지 한 개를 받아 에이전트(GPT)와 주고받는 전체 흐름을 처리하는 함수.

    동작 순서:
    1) 이전 대화 기록(conversation_history)에 이번 사용자 메시지를 이어붙인다.
    2) GPT에게 커스텀 함수(predict_price_iphone, detect_anomaly) +
       호스팅 툴(web_search, file_search)을 함께 제공하며 응답을 요청한다.
    3) 응답에 "함수 호출 요청"이 있으면 -> 실제 파이썬 함수를 실행하고,
       그 결과를 다시 대화 기록에 추가해서 GPT에게 재요청한다. (함수 호출이 없을 때까지 반복)
    4) 함수 호출 요청이 더 이상 없으면 -> 최종 자연어 답변과 누적된 대화 기록을 반환한다.

    Args:
        user_message (str): 사용자가 입력한 질문/메시지
        conversation_history (list): 지금까지 쌓인 대화 기록 (Responses API 형식의 input 리스트)

    Returns:
        tuple: (최종 답변 텍스트(str), 갱신된 대화 기록(list))
    """
    # 최초 호출 시 conversation_history가 None으로 들어올 수 있으므로 빈 리스트로 초기화
    if conversation_history is None:
        conversation_history = []

    # 커스텀 툴(predict_price_iphone, detect_anomaly)과 호스팅 툴(web_search, file_search) 합치기
    all_tools = tools + [
        {
            "type": "web_search_preview",
            # 검색 컨텍스트를 줄여서 검색 횟수와 깊이를 제한 (비용/속도 절약)
            "search_context_size": "low",
        },
        {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]},
    ]

    # 대화 기록 (에이전트 응답, 함수 실행 결과가 계속 쌓임)
    if not conversation_history:
        # 최초 대화라면 시스템 프롬프트(developer 역할)를 맨 앞에 넣어준다
        input_messages = [
            # 이게 없으면 GPT가 함수 안 쓰고 알아서 답함
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    else:
        # 이미 대화가 진행 중이라면 기존 기록 뒤에 새 사용자 메시지만 추가
        input_messages = conversation_history + [{"role": "user", "content": user_message}]

    # 반복 횟수 확인용 (디버깅 로그에 몇 번째 API 호출인지 표시)
    round_num = 0

    # 에이전트가 함수 호출을 계속 요청할 경우 -> 무한 루프로 처리
    # (함수 호출 -> 결과 전달 -> 다시 GPT 호출 -> 필요하면 또 함수 호출... 을 반복)
    while True:
        round_num += 1
        print(f"\n[디버그] ===== {round_num}번째 API 호출 시작 =====")
        print(f"[디버그] 지금까지 쌓인 input_messages 개수: {len(input_messages)}")

        # GPT에게 현재까지의 대화 기록 + 사용 가능한 툴 목록을 전달하여 응답 생성 요청
        response = client.responses.create(
            model="gpt-5.5",          # 모델 5 -> 5.5
            tools=all_tools,
            input=input_messages,
        )

        # response.output에 있는 모든 항목의 타입을 확인 (예: message, function_call 등)
        print(f"[디버그] 이번 응답에 포함된 항목 타입들: {[item.type for item in response.output]}")

        # 응답에 커스텀 함수 호출 요청이 있었는지 필터링
        function_calls = [item for item in response.output if item.type == "function_call"]

        # 함수 호출 요청이 하나도 없다면 -> GPT가 최종 답변을 텍스트로 생성한 것이므로 루프 종료
        if not function_calls:
            print(f"[디버그] -> 함수 호출 요청 없음. GPT가 자체 지식/이전 함수 결과로 최종 텍스트 답변 생성함.")
            print(f"[디버그] -> 이번 응답의 output_text 앞부분: {response.output_text[:300]}...")
            # 최종 답변도 기록에 추가 (다음 turn에서 문맥으로 활용하기 위함)
            input_messages += response.output
            return response.output_text, input_messages

        print(f"[디버그] -> 이번 응답에서 호출 요청된 함수: {[call.name for call in function_calls]}")

        # 이번 응답(함수 호출 요청 포함)을 대화 기록에 추가
        input_messages += response.output

        # 요청된 함수들을 하나씩 실제로 실행
        for call in function_calls:
            # 함수 이름으로 실제 파이썬 함수 객체 조회
            func = AVAILABLE_FUNCTIONS.get(call.name)
            # GPT가 넘겨준 인자(JSON 문자열)를 파이썬 dict로 변환
            args = json.loads(call.arguments)
            print(f"[디버그]    실행 중: {call.name}({args})")

            # 함수가 존재하면 실행, 없으면 에러 딕셔너리 반환
            result = func(**args) if func else {"error": f"알 수 없는 함수: {call.name}"}
            print(f"[디버그]    실행 결과: {result}")

            # 함수 실행 결과를 대화 기록에 추가 (에이전트가 다음 턴에 이 결과를 보고 답변 생성)
            input_messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,          # 어떤 함수 호출 요청에 대한 결과인지 매칭용 ID
                "output": json.dumps(result, ensure_ascii=False),  # 한글 깨짐 방지(ensure_ascii=False)
            })

        print(f"[디버그] -> 함수 실행 결과를 input_messages에 추가함. 다음 반복에서 이 결과를 GPT에게 다시 보여줌.")
        # while 루프의 처음으로 돌아가 GPT를 다시 호출 (함수 결과를 바탕으로 추가 답변 또는 추가 함수 호출)

# ==========================================================
# 메인 실행부
# ==========================================================
# storage_gb, condition의 enum 값은 지금 내가 일반적인 아이폰 스토리지/상태 등급 기준으로 임의로 넣은 것. 
# 실제 학습 데이터(train.py/전처리)에서 쓰는 카테고리 값이랑 반드시 똑같아야 하니까, 팀원한테 정확한 값 목록 확인해서 여기 맞춰야 함.

if __name__ == "__main__":
    # 대화 기록 누적 (매 turn마다 orchestrate()에 넘겨서 문맥 유지)
    history = []
    
    # 사용자가 "exit"을 입력할 때까지 계속 질문을 받는 반복 루프
    while True:
        user_message = input("질문을 입력하세요 (예: 아이폰16프로 256GB Used 120만원이면 괜찮아?): ")

        # "exit" 입력 시 종료 
        if user_message.strip().lower() == "exit":
            break

        # history 계속 이어서 전달 -> orchestrate()가 새 답변과 갱신된 history를 반환
        answer, history = orchestrate(user_message, history)

        print("==================== 답변 ====================")
        print(answer)