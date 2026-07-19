"""
Streamlit UI

역할:
- 중고가 예측이 필요한 노트북/아이폰 정보 입력
- 종목에 맞는 agent의 모델을 통해 중고가 예측
- 챗봇 형태로 추가 질의 응답 및 예측 결과 응답
"""

import streamlit as st
import agent_iphone as iphone
import agent_laptop as laptop


# -----------------------------
# 초기 설정
# -----------------------------
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

if "device_type" not in st.session_state:
    st.session_state.device_type = "노트북"

if "loading" not in st.session_state:
    st.session_state.loading = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "orchestrate_history" not in st.session_state:
    st.session_state.orchestrate_history = []

if "started" not in st.session_state:
    st.session_state.started = False

if "searching" not in st.session_state:
    st.session_state.searching = False

# 첫 질문에 첨부된 이미지(노트북 사양 화면 등) 저장용
# 이미지 파일의 실제 바이트 데이터 (사진 그 자체)
if "first_image_bytes" not in st.session_state:
    st.session_state.first_image_bytes = None
 
# "image/png" 같은 문자열 (파일 형식 정보)
if "first_image_type" not in st.session_state:
    st.session_state.first_image_type = None

# -----------------------------
# Agent 관리
# -----------------------------
AGENTS = {
    "아이폰": iphone,
    "노트북": laptop
}

def get_agent():
    return AGENTS.get(
        st.session_state.device_type,
        laptop
    )

# -----------------------------
# Chat 관리
# -----------------------------
# 메세지 기록 저장
def add_message(role, content):
    st.session_state.messages.append(
        {
            "role": role,
            "content": str(content)
        }
    )
# 사용자 질문 받아서 Agent 호출
def ask_agent(agent,user_message, history):
    if agent is laptop:
        return agent.orchestrate(
            user_message,
            history,
            image_bytes=st.session_state.get("first_image_bytes"),
            image_media_type=st.session_state.get("first_image_type") or "image/png",
        )
    return agent.orchestrate(user_message, history)

# input placeholer 
def get_chat_placeholder():
    if st.session_state.loading:
        return "🤖 답변 생성 중입니다..."

    if len(st.session_state.messages) == 0:

        if st.session_state.device_type == "아이폰":
            return "예 : 아이폰 17 화이트 256GB 가격이 얼마인가요?"

        else:
            return "예 : 델 래티튜드 i5 16GB 512GB SSD 중고 50만원이면 괜찮나요?"

    # 두 번째 질문부터
    return "추가적인 질문을 입력하세요."

# 채팅 기록 초기화
def reset_chat(device):

    st.session_state.device_type = device
    st.session_state.messages = []
    st.session_state.orchestrate_history = []
    st.session_state.started = False

def first_message():
    # 기존 코드에서 사용하던 부분이라 일단 남겨둠 - 추후에 바꿀수도 있음 // 아이폰이랑 노트북 구분하기 위함
    with st.form("first_question"):
        if st.session_state.device_type == "아이폰":
            first_prompt = st.text_input("첫번째 질문",placeholder="예 : 아이폰 17 화이트 256gb는 얼마인가요?",label_visibility="collapsed")
        else:
            first_prompt = st.text_input("첫번째 질문",placeholder="예 : 델 래티튜드 i5 16GB 512GB SSD Used 50만원이면 괜찮아?",label_visibility="collapsed")
            # 노트북 사양 화면 캡처 / 모델명 라벨 사진 업로드 (선택)
            uploaded_image = st.file_uploader(
                "사양 화면 캡처나 모델명 라벨 사진이 있으면 함께 올려주세요 (선택)",
                type=["png", "jpg", "jpeg"],
            )

        submit = st.form_submit_button("검색", use_container_width=True, disabled=st.session_state.searching)

    if submit:
        st.session_state.searching = True
        st.session_state.first_prompt = first_prompt
        # 업로드된 이미지를 세션에 bytes로 저장
        if uploaded_image is not None:
            st.session_state.first_image_bytes = uploaded_image.read()
            st.session_state.first_image_type = uploaded_image.type
        else:
            st.session_state.first_image_bytes = None
            st.session_state.first_image_type = None
        st.rerun()

    # searching 상태에서 실제 실행
    if st.session_state.searching:

        first_prompt = st.session_state.first_prompt

        st.session_state.messages.append({
            "role": "user",
            "content": first_prompt
        })

        st.session_state.orchestrate_history = []

        with st.spinner("🤖 답변을 생성 중입니다..."):
            agent_module = get_agent()
            # result, st.session_state.orchestrate_history = agent_module.orchestrate(
            #     first_prompt,
            #     st.session_state.orchestrate_history
            # )
            # call_orchestrate()로 통일 - 노트북일 때만 이미지 같이 전달됨
            result, st.session_state.orchestrate_history = ask_agent(
                agent_module,
                first_prompt,
                st.session_state.orchestrate_history,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": str(result)
        })

        st.session_state.started = True

        # 완료 후 다시 활성화
        st.session_state.searching = False
        st.rerun()

# -----------------------------
# 화면 구성
# -----------------------------
# Header
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

# Sidebar
with st.sidebar:

    st.markdown(
        """
        # 🤖 Price AI
        """
    )

    st.divider()

    st.markdown(
        """
        1. 제품 정보를 입력하세요.

        > ★ TIP ★ 자세히 적을수록 정확한 시세를 알 수 있습니다.

        2. AI가 적정 가격을 분석합니다.

        3. 추가 질문을 할 수 있습니다.
        """
    )


    st.write("### 제품 선택")


    if st.button(
        "💻 노트북",
        use_container_width=True
    ):
        reset_chat("노트북")
        st.rerun()


    if st.button(
        "📱 아이폰",
        use_container_width=True
    ):
        reset_chat("아이폰")
        st.rerun()

# Chat 화면
    # 기존 대화 출력
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


 # 사용자에게 받은 질문 처리
first_chat = not st.session_state.started

if first_chat:
    first_message()
else:
# 입력 단계
    prompt = st.chat_input(
    get_chat_placeholder()
    )


    if prompt:

        add_message(
        "user",
        prompt
        )

        st.session_state.pending_prompt = prompt
        st.session_state.loading = True

        st.rerun()

# 답변 생성 단계
if st.session_state.loading:
    try:
        with st.spinner("🤖 답변 생성 중..."):
            agent_module = get_agent()
            result, st.session_state.orchestrate_history = ask_agent(
                agent_module,
                st.session_state.pending_prompt,
                st.session_state.orchestrate_history,
            )
        add_message(
            "assistant",
            result
        )
    except Exception as e:
        st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
    finally:
        st.session_state.loading = False
        st.session_state.pending_prompt = None
        st.rerun()

