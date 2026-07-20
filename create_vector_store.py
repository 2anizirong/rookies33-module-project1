"""
create_vector_store.py

최초 1회 실행하는 파일입니다.
Vector Store를 생성하고,
생성된 VECTOR_STORE_ID를 발급받습니다.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 환경 변수 로드
# ==========================================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ==========================================
# Vector Store 생성
# ==========================================

print("Vector Store 생성 중..")

vector_store = client.vector_stores.create(
    name= "종합 중고 구매 가이드"
)

# ==========================================
# 생성된 Vector Store ID 출력
# ==========================================

print("vector store 생성 완료")
print(f"VECTOR_STORE_ID : {vector_store.id}")

print("\n.env 파일에 아래 내용을 추가하세요.\n")
print(f"VECTOR_STORE_ID={vector_store.id}")