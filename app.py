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
import agent_iphone as iphone
import agent_laptop as laptop

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="AI 전자기기 적정가격 예측 어시스턴트",layout="wide")

# st.markdown("<br><br><br><br>", unsafe_allow_html=True)

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
# css 파일 로드 
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
load_css("styles/style.css")

def get_agent():
    if st.session_state.device_type == "아이폰":
        return iphone
    return laptop


def first_message ():
    # 기존 코드에서 사용하던 부분이라 일단 남겨둠 - 추후에 바꿀수도 있음 // 아이폰이랑 노트북 구분하기 위함
    with st.form("first_question"):
        if st.session_state.device_type == "아이폰":
            first_prompt = st.text_input("첫번째 질문",placeholder="예 : 아이폰 17 화이트 256gb는 얼마인가요?",label_visibility="collapsed")
        else:
            first_prompt = st.text_input("첫번째 질문",placeholder="예 : 델 래티튜드 i5 16GB 512GB SSD Used 50만원이면 괜찮아?",label_visibility="collapsed")

        submit = st.form_submit_button("검색", use_container_width=True)

    if first_prompt and submit:
        st.session_state.messages.append({"role":"user","content":first_prompt})
        st.session_state.orchestrate_history = []

        with st.spinner("🤖 답변을 생성 중입니다..."):
            agent_module = get_agent()
            result, st.session_state.orchestrate_history = agent_module.orchestrate(
                first_prompt,
                st.session_state.orchestrate_history
            )

        st.session_state.messages.append({
            "role":"assistant",
            "content": str(result)
            })
        
        st.session_state.started = True
        st.rerun()


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

if "loading" not in st.session_state:
    st.session_state.loading = False

first_chat = not st.session_state.started

if first_chat:
    
    first_message()

    if st.session_state.show_detail:

        if st.session_state.device_type is None:

            col1, col2, col3, col4 = st.columns([3,2,2,3])
            with col2:
                if st.button("📱 아이폰", use_container_width=True):
                    st.session_state.device_type = "아이폰"
                    # agent_iphone.out
                    st.rerun()
            with col3:
                if st.button("💻 노트북", use_container_width=True):
                    st.session_state.device_type = "노트북"
                    # agent_laptop
                    st.rerun()
    

# -----------------------------
# 채팅 화면
else:

    # 기존 대화 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 예측 결과  테스트용 데이터
 # if st.session_state.predicted:
        # st.success("예상 시세 : 80만원 ~ 90만원")
        # st.metric("적정가격","850,000원")
        # st.info("가격 비교 결과  :")   

    # if st.session_state.loading:
    #     st.info("🤖 답변을 생성 중입니다. 잠시만 기다려주세요...")

    # else:
    prompt = st.chat_input("추가적인 질문을 입력하세요.")

    if prompt:
        st.session_state.loading = True

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("🤖 답변을 생성 중입니다..."):
            agent_module = get_agent()
            result, st.session_state.orchestrate_history = agent_module.orchestrate(
                prompt,
                st.session_state.orchestrate_history
            )

        print("APP:::", result)

        st.session_state.messages.append({
            "role": "assistant",
            "content": str(result)
        })

        st.session_state.loading = False
        st.rerun()

# 상세검색 버튼
# col_btn, _ = st.columns([1, 8])

# with col_btn:
#     if st.button("상세검색", use_container_width=True):
#         st.session_state.show_detail = not st.session_state.show_detail

#         # 펼칠 때마다 기기 선택 초기화
#         if st.session_state.show_detail:
#             st.session_state.device_type = None


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

with st.sidebar:
    st.sidebar.markdown("""
    # 🤖 Price AI
    """)
    # st.title("Laptop Advisor")
    st.divider()

    st.write("### 사용 방법")
    st.write("""
    1. 제품 정보를 입력하세요.
    2. AI가 적정 가격을 분석합니다.
    3. 추가 질문을 할 수 있습니다.
    """)

    st.write("### 상세 검색")
    # if st.session_state.device_type == "아이폰" :
    if st.button("💻 노트북", use_container_width=True):
            st.session_state.device_type = "노트북"
            st.session_state.started = False
            st.session_state.messages = []
            st.rerun()
    # else :
    if st.button("📱 아이폰", use_container_width=True):
            st.session_state.device_type = "아이폰"
            st.session_state.started = False
            st.session_state.messages = []
            st.rerun()



