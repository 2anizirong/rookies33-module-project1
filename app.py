"""
Streamlit UI (UI 파트: 원종현)

역할:
- 중고 노트북 정보 입력 (브랜드/모델/RAM/저장공간/상태/판매가격)
- agent.run_pipeline() 호출 -> 예측 가격범위 + 분류 + AI 설명 표시
- 챗봇 형태로 추가 질의 응답
"""

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils.forms import iphone_form, laptop_form
from utils.validator import validate_iphone, validate_laptop

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="AI 전자기기 적정가격 예측 어시스턴트",layout="wide")

# css 파일 로드 
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
load_css("styles/style.css")

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "predicted" not in st.session_state:
    st.session_state.predicted = False

if "started" not in st.session_state:
    st.session_state.started = False

if "show_detail" not in st.session_state:
    st.session_state.show_detail = False

if "device_type" not in st.session_state:
    st.session_state.device_type = None

first_chat = not st.session_state.started

# -----------------------------
# 첫 화면  첫번재 화면에서 AI가 답변해주는 소스필요한지?
# -----------------------------
if first_chat:

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center;">
            <h1>중고제품 가격비교 AI</h1>
            <h4 style="color:gray;">
                구매하려는 전자기기의 가격을 알려드립니다.
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, _, _ = st.columns([15,1,1])
    with col1:
        first_prompt = st.text_input("",placeholder="예 : 아이폰 17 화이트 256gb는 얼마인가요?")
    #with col2:
        #upload=st.file_uploader("",label_visibility="collapsed")

    if first_prompt:
        st.session_state.started = True
        st.session_state.messages.append({"role":"user","content":first_prompt})
        st.session_state.messages.append({
                "role":"assistant",
                "content":"안녕하세요 😊 노트북 정보를 입력해주세요."})
        st.rerun()

# -----------------------------
# 채팅 화면
else:

    # 기존 대화 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 예측 결과  테스트용 데이터
    if st.session_state.predicted:
        st.success("예상 시세 : 80만원 ~ 90만원")
        st.metric("적정가격","850,000원")
        st.info("가격 비교 결과  :")


# 상세검색 버튼
col_btn, _ = st.columns([1, 8])

with col_btn:
    if st.button("상세검색", use_container_width=True):
        st.session_state.show_detail = not st.session_state.show_detail

        # 펼칠 때마다 기기 선택 초기화
        if st.session_state.show_detail:
            st.session_state.device_type = None


# -----------------------------
# 상세검색
# -----------------------------
if st.session_state.show_detail:

    if st.session_state.device_type is None:

        col1, col2, col3, col4 = st.columns([3,2,2,3])
        with col2:
            if st.button("📱 아이폰", use_container_width=True):
                st.session_state.device_type = "아이폰"
                st.rerun()
        with col3:
            if st.button("💻 노트북", use_container_width=True):
                st.session_state.device_type = "노트북"
                st.rerun()
    else :
        with st.form("input_form"):
            if st.session_state.device_type == "아이폰":
                model, storage, condition, color, input_price = iphone_form()
            else :
                brand,model,condition,os_type,ram,ssd,input_price =laptop_form()
            btn_left, btn_right = st.columns([9,1])
            with btn_right :
                submit = st.form_submit_button("조회", use_container_width=True)
        
        #조회버튼 유효성 검사
        if submit:
            errors = []
            if st.session_state.device_type =='아이폰':
                errors =validate_iphone(model, storage, condition)
            elif st.session_state.device_type =='노트북':
                errors =validate_laptop(brand,model,condition)

            # 오류가 있으면 출력
            if errors:
                for error in errors:
                    st.error(error)
            # API 콜 
            else:
                st.session_state.predicted = True
                st.session_state.show_detail = False
                st.session_state.device_type = None
                # OPENAPI 질문 물어보기 : 모델 호출
                # result = predict(model, storage, condition, color)

                st.rerun()

# 사이드 바 
with st.sidebar:

    st.title("Laptop Advisor")
    st.divider()
    st.write("### 메뉴")


if st.session_state.started:
    prompt = st.chat_input("무엇이든 물어보세요")

    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        st.session_state.messages.append({"role":"assistant","content":"(UI 테스트용 답변입니다.)"})
        st.rerun()



