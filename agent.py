"""
agent.py

<필요 함수>
predict_price(), detect_anomaly(), search_market_price() - web search,
search_buying_guide() - file search, generate_result()
"""

# ==========================================================
# 모듈 import (전부 파일 최상단에 모음)
# ==========================================================
import os

import joblib
from dotenv import load_dotenv
from openai import OpenAI

from preprocess import preprocess_input    # 팀원 구현 예정, 함수명/반환형태 확인 필요


# ==========================================================
# 상수 / 클라이언트 / 모델 로드
# ==========================================================

# .env 파일에서 환경변수 로드 (OPENAI_API_KEY 등)
load_dotenv()

# OpenAI API Client 생성 - 환경변수에서 키를 불러와 사용 (하드코딩 금지)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# File Search에서 사용할 Vector Store ID (이것도 .env에서 관리하는 걸 추천)
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# 모델은 앱 실행 시 한 번만 로드
MODEL_PATH = "models/price_model.pkl"
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
feature_columns = model_data["feature_columns"]


# ==========================================================
# Web Search : 최신 중고 시세 검색
# ==========================================================
def search_market_price(product_name: str) -> str:

    # GPT에게 전달할 프롬프트
    prompt = f"""
    '{product_name}'의 최신 중고 거래 시세를 검색해주세요.

    다음 내용을 포함하여 알려주세요.

    1. 최저 거래가
    2. 평균 거래가
    3. 최고 거래가
    4. 최근 시세 동향
    """

    # OpenAI Web Search 실행
    response = client.responses.create(
        model="gpt-5",
        tools=[
            {
                "type": "web_search_preview"
            }
        ],
        input=prompt
    )

    # 검색 결과 반환
    return response.output_text


# ==========================================================
# File Search : 휴대폰 구매 가이드 검색
# ==========================================================
def search_buying_guide(product_name: str) -> str:

    # GPT에게 전달할 프롬프트
    prompt = f"""
    내부 구매 가이드 문서를 참고하여
    '{product_name}' 구매 시 확인해야 하는 사항을 알려주세요.

    다음 항목에 대해서만 알려주세요.

    1. 저장 용량 (128GB / 256GB / 512GB)
    2. 배터리 성능(효율)
    """

    # OpenAI File Search 실행
    response = client.responses.create(
        model="gpt-5",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID]
            }
        ],
        input=prompt
    )

    # 검색 결과 반환
    return response.output_text


# predict_price()
# ML 모델 호출
# 입력받은 아이폰 정보를 ML 모델에 전달하여 적정 중고가를 반환한다.
# 아이폰16프로 120만원이면 괜찮아??
def predict_price(      # 제품명, 저장용량, 제품 상태를 이용하여 적정 중고가를 예측하는 함수
    title: str,         
    storage_gb: int,   
    condition: str      
) -> dict:
    
    # 입력 데이터 전처리
    # TODO: preprocess_input()이 title/storage_gb/condition을
    #       feature_columns(800+ 컬럼) 구조에 맞는 원-핫 벡터로 변환해줘야 함
    #       (팀원 확인 필요: title 그대로 넣는지, model_family로 정규화해서 넣는지)
   
    features = preprocess_input(
        title=title,
        storage_gb=storage_gb,
        condition=condition
    )
    
    # 모델 예측 실행
    predicted_price = model.predict(features)[0]

    return {
        "predicted_price": int(predicted_price),
        "residual_std": model_data["residual_std"]  # detect_anomaly()에서 사용
    }


# detect_annomaly()
def detect_anomaly(
    predicted_price: int,
    selling_price: int,
    residual_std: float   # pkl에서 로드된 model_data["residual_std"] 사용 권장
) -> dict:
    # 예측가격과 판매가격을 비교하여 이상 매물 여부를 판단한다

    # 방어 코드: 예측가 또는 판매가가 유효하지 않으면 비교 자체가 불가능
    if predicted_price <= 0 or selling_price <= 0:
        return {
            "status": "ERROR",
            "message": "예측 가격 또는 판매 가격이 유효하지 않아 비교할 수 없습니다.",
            "difference": None,
            "difference_percent": None
        }

    difference = selling_price - predicted_price
    percent = (difference / predicted_price) * 100

    # residual_std 기준: 모델 예측 오차의 표준편차 대비 몇 배 벗어났는지로 판단
    deviation = abs(difference) / residual_std

    if deviation < 1:
        status = "NORMAL"
        message = "정상 거래 범위입니다."
    elif deviation < 2:
        status = "WARNING"
        message = "시세와 다소 차이가 있습니다."
    else:
        status = "DANGER"
        message = "허위 매물 가능성이 있으니 주의하세요."

    return {
        "status": status,
        "message": message,
        "difference": difference,
        "difference_percent": round(percent, 2)
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
        "description": "predict_price()로 얻은 예측 가격(및 모델 오차 정보)과 사용자가 제시한 판매 가격을 비교하여 정상/주의/위험 여부를 판단한다.",
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


# ==========================================================
# 메인 실행부
# ==========================================================

# input_message = [
#     {
#         "role": "user",
#         "content": "아이폰16 프로 256GB 상태 Excellent인데 120만원이면 괜찮아?"
#     }
# ]

result = predict_price(
    title="iPhone 16 Pro",
    storage_gb=256,
    condition="Very Good - Refurbished"
)
print(result)

# storage_gb, condition의 enum 값은 지금 내가 일반적인 아이폰 스토리지/상태 등급 기준으로 임의로 넣은 것. 
# 실제 학습 데이터(train.py/전처리)에서 쓰는 카테고리 값이랑 반드시 똑같아야 하니까, 팀원한테 정확한 값 목록 확인해서 여기 맞춰야 함.

if __name__ == "__main__":
    # 사용자가 휴대폰 모델명을 입력
    product_name = input("휴대폰 모델명을 입력하세요 : ")

    # 최신 중고 시세 검색
    market_result = search_market_price(product_name)

    # 구매 가이드 검색
    guide_result = search_buying_guide(product_name)


    # ==========================================================
    # 결과 출력
    # ==========================================================

    print("\n========== 최신 중고 시세 ==========")
    print(market_result)

    print("\n========== 휴대폰 구매 가이드 ==========")
    print(guide_result)