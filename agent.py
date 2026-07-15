"""
agent.py

<필요 함수>
predict_price(), detect_anomaly(): predict.py 에서 import만 하기
search_market_price() - web search
search_buying_guide() - file search
generate_result()
"""

# ==========================================================
# 모듈 import (전부 파일 최상단에 모음)
# ==========================================================
import os

from dotenv import load_dotenv
from openai import OpenAI

from predict import predict_price, detect_anomaly   # predict.py에서 가져오기 
import json

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

# 모델은 앱 실행 시 한 번만 로드 -> 이것도 predict.py에서 관여하기 
# MODEL_PATH = "models/price_model.pkl"
# model_data = joblib.load(MODEL_PATH)
# model = model_data["model"]
# feature_columns = model_data["feature_columns"]

# 실제 함수 이름으로 함수 객체 매핑하기 (agent가 호출 요청하면 여기서 실행)
AVAILABLE_FUNCTIONS = {
    "predict_price": predict_price,
    "detect_anomaly": detect_anomaly
}

# ==========================================================
# Function Calling용 tools 정의
# ==========================================================
tools = [
    {
        "type": "function",
        "name": "predict_price",
        "description": "입력받은 아이폰 정보를 기반으로 머신러닝 모델을 호출하여 적정 중고가를 예측한다.",
        "strict": True,
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
                    "enum": [64, 128, 256, 512, 1024, 2048]
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
            "required": ["title", "storage_gb", "condition"],
            "additionalProperties": False
        }
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
                    "description": "predict_price 함수가 반환한 예측 적정가(원)"
                },
                "selling_price": {
                    "type": "number",
                    "description": "사용자가 제시한 실제 판매 가격(원)"
                },
                "residual_std": {
                    "type": "number",
                    "description": "predict_price 함수가 반환한 모델 오차의 표준편차"
                }
            },
            "required": ["predicted_price", "selling_price", "residual_std"],
            "additionalProperties": False
        }
    }          
]

# 사용자 메시지를 받아서
# 에이전트한테 툴이랑 같이 해서 전달
# 에이전트가 커스텀툴 (2개_predict_price랑 detect_anomaly) 호출 요청하면 실행해서 결과 에이전트에게 다시 돌려주기
# 에이전트가 더 이상 함수 호출이 필요없으면 자연어로 출력해서 사용자에게 반환
# refactor: 이전까지의 대화 기록 누적
def orchestrate(user_message: str, conversation_history: list) -> tuple:
    if conversation_history is None:
        conversation_history = []

    # 커스텀 툴이랑 호스팅 툴 전달
    all_tools = tools + [
        {"type": "web_search_preview"},
        # {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]},
    ]

    # 대화 기록 (에이전트 응답, 함수 실행 결과가 계속 쌓임)
    input_messages = conversation_history + [
        {
            "role": "user",
            "content": user_message
        }
    ]

    # 에이전트가 함수 호출을 계속 요청할 경우 -> 무한 루프로 처리
    while True:
        response = client.responses.create(
            model="gpt-5",          # 모델은 위에 쓴 모델이랑 통일
            tools=all_tools,
            input=input_messages
        )

        # 웅답에 커스텀 함수 호출 부탁이 있었는지
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            # 최종 답변도 기록에 추가
            input_messages += response.output
            return response.output_text, input_messages
        
        # 이전 응답에 대화 기록 추가  
        input_messages += response.output

        # 이제 하나씩 실행
        for call in function_calls:
            func = AVAILABLE_FUNCTIONS.get(call.name)
            result = func(**json.loads(call.arguments)) if func else {"error": f"알 수 없는 함수: {call.name}"}
            
            # 함수 실행 결과를 대화 기록에 추가 (에이전트가 다음 턴에 이 결과를 보고 답변 생성)
            input_messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })


# ==========================================================
# 메인 실행부
# ==========================================================
# storage_gb, condition의 enum 값은 지금 내가 일반적인 아이폰 스토리지/상태 등급 기준으로 임의로 넣은 것. 
# 실제 학습 데이터(train.py/전처리)에서 쓰는 카테고리 값이랑 반드시 똑같아야 하니까, 팀원한테 정확한 값 목록 확인해서 여기 맞춰야 함.

if __name__ == "__main__":
    # 대화 기록 누적
    history = []
    
    while True:
        user_message = input("질문을 입력하세요 (예: 아이폰16프로 256GB Used 120만원이면 괜찮아?): ")

        # "exit" 입력 시 종료 
        if user_message.strip().lower() in ("exit"):
            break

        # history 계속 이어서 전달
        answer, history = orchestrate(user_message, history)

        print("==================== 답변 ====================")
        print(answer)