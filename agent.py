"""
OpenAI Agent SDK 오케스트레이션 (Agent 파트: 장한수, 김민규)

역할:
- 메인(오케스트레이터) 에이전트 정의
- predict_price_range()를 function tool로 등록
- 필요 시 web_search / file_search 호스팅 툴 연결
- 사용자 추가 질문에 Function Calling으로 응답
"""

# from agents import Agent, Runner, function_tool
from predict import predict_price_range


# @function_tool
def price_prediction_tool(features: dict) -> dict:
    """에이전트가 호출할 가격 예측 툴"""
    return predict_price_range(features)


def build_main_agent():
    """
    메인 에이전트 생성
    - tools=[price_prediction_tool, web_search, file_search] 형태로 연결
    - system prompt: 가격 판단 근거를 설명하되 사기 여부는 판정하지 않음
    """
    raise NotImplementedError


def run_pipeline(user_input: dict):
    """
    1) price_prediction_tool 호출 -> 적정가격/가격범위/분류
    2) 에이전트가 자연어로 판단 근거 설명
    3) 사용자의 추가 질문에 Function Calling으로 응답
    """
    raise NotImplementedError

# <필요함수>
# predict_price(), detect_anomaly(), search_market_price()- web search, search_buying_guide()- file search, generate_result()
#

# web search/file search 코딩

from openai import OpenAI



# OpenAI API Client 생성
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# File Search에서 사용할 Vector Store ID
# (휴대폰 구매 가이드 PDF 업로드 후 생성되는 ID)
VECTOR_STORE_ID = "YOUR_VECTOR_STORE_ID"


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


# ==========================================================
# 메인 실행부
# ==========================================================

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