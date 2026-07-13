"""
Streamlit UI (UI 파트: 원종현)

역할:
- 중고 노트북 정보 입력 (브랜드/모델/RAM/저장공간/상태/판매가격)
- agent.run_pipeline() 호출 -> 예측 가격범위 + 분류 + AI 설명 표시
- 챗봇 형태로 추가 질의 응답
"""

import streamlit as st
from agent import run_pipeline

st.set_page_config(page_title="AI 중고 노트북 적정가격 예측 어시스턴트", layout="wide")
st.title("AI 기반 중고제품 적정가격 예측 및 이상가격 탐지 서비스")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input("브랜드")
        model = st.text_input("모델명")
        ram = st.number_input("RAM (GB)", min_value=0, step=1)
    with col2:
        storage = st.number_input("저장공간 (GB)", min_value=0, step=1)
        condition = st.selectbox("상태", ["신품급", "상", "중", "하"])
        input_price = st.number_input("판매가격 (원)", min_value=0, step=1000)

    submitted = st.form_submit_button("적정가격 확인하기")

if submitted:
    features = {
        "brand": brand, "model": model, "ram": ram,
        "storage": storage, "condition": condition, "input_price": input_price,
    }
    # result = run_pipeline(features)
    st.info("agent.py의 run_pipeline() 구현 후 결과가 여기에 표시됩니다.")

st.divider()
st.subheader("추가로 궁금한 점을 물어보세요")
st.chat_input("예: 이 가격대 노트북 중에 더 좋은 옵션 있어?")
