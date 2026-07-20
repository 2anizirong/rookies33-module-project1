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
# 새로운 PDF가 추가될 경우 docs 폴더에서 관리합니다.
PDF_PATH = "docs/노트북 중고 구매 가이드.pdf" 


# PDF 업로드
print("PDF 업로드 중...")

uploaded_file = client.files.create(
    file=open(PDF_PATH, "rb"),
    purpose="assistants"
)

print("파일 업로드 완료")
print("File ID :", uploaded_file.id)


# Vector Store에 PDF 등록
print("Vector Store 등록 중...")

client.vector_stores.files.create(
    vector_store_id=VECTOR_STORE_ID,
    file_id=uploaded_file.id
)


print("PDF 등록 완료")
