import os
from dotenv import load_dotenv
from openai import OpenAI


# 환경 변수 로드
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")


# 업로드할 PDF 경로
PDF_PATH = "docs/아이폰 중고 구매 가이드.pdf" # 추가 pdf파일이 생길 경우 관리 편리성을 위해 docs파일에서 관리


# PDF 업로드
print("PDF 업로드 중...")

uploaded_file = client.files.create(
    file=open(PDF_PATH, "rb"),
    purpose="assistants"
)

print("파일 업로드 완료")
print("File ID :", uploaded_file.id)


# Vector Store에 PDF 연결
print("Vector Store 연결 중...")

client.vector_stores.files.create(
    vector_store_id=VECTOR_STORE_ID,
    file_id=uploaded_file.id
)


print("PDF 등록 완료")
