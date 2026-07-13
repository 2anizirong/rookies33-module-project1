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
