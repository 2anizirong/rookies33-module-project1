# AI 기반 중고제품 적정가격 예측 및 이상가격 탐지 서비스

SK쉴더스 루키즈 33기 모듈프로젝트1 (2026.07.13 ~ 07.20)

## 프로젝트 개요
중고 노트북의 브랜드/모델/사양/상태/판매가격을 입력하면, 머신러닝으로
예상 적정가격과 가격 범위를 예측하고, 입력 가격을 저가/적정가/고가로
분류한 뒤 OpenAI Agent SDK가 판단 근거를 자연어로 설명하는 서비스입니다.

데이터: eBay Laptops & Netbooks Sales (Unclean), Kaggle

## 폴더 구조
```
.
├── data/
│   ├── raw/                # 원본 eBay CSV
│   └── processed/          # 전처리 완료 데이터
├── notebooks/               # EDA, 전처리, 모델링 실험용 노트북
├── models/                  # 학습된 모델(.pkl) 저장
├── docs/                    # 보고서, 발표자료
├── preprocess.py            # 데이터 전처리
├── train.py                 # 모델 학습
├── predict.py                # 가격 예측 함수 (Agent 커스텀 툴)
├── agent.py                   # OpenAI Agent SDK 오케스트레이션
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env.example
└── .gitignore
```

## 핵심 기능 (MVP)
1. 적정가격 예측 (ML 회귀 모델)
2. 가격 이상치 판별 (저가 / 적정가 / 고가 분류)
3. AI 가격 판단 설명 (LLM)
4. 챗봇을 통한 추가 질의 응답 (Function Calling)

## 역할 분담

| 파트 | 담당자 | 담당 파일 |
|---|---|---|
| 데이터 전처리 | 김상현, 김혜빈 | `preprocess.py`, `data/` |
| 모델 학습 | 김윤호, 김이안 | `train.py`, `predict.py`, `models/` |
| OpenAI API / Agent | 장한수, 김민규 | `agent.py` |
| Streamlit UI | 원종현 | `app.py` |

## ⭐⭐⭐ 실행 방법
```bash
pip install -r requirements.txt
cp .env.example .env      # OPENAI_API_KEY 입력

python preprocess.py       # 1. 데이터 전처리
python train.py             # 2. 모델 학습 -> models/price_model.pkl 생성
streamlit run app.py         # 3. 서비스 실행
```

## ⭐⭐⭐ 주의사항
- `.env` 파일은 절대 git에 커밋하지 않습니다 (API 키 노출 방지).
- 노트북 실험 결과 중 최종 로직만 루트의 `.py` 파일로 옮겨서 정리합니다.

## 협업 컨벤션

### 커밋 메시지 규칙

`태그: 작업 내용` 형식으로 작성합니다.

| 태그 | 설명 |
|---|---|
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `data` | 데이터 추가/수정/전처리 관련 |
| `model` | 모델 학습/튜닝 관련 |
| `docs` | 문서 수정 (README, 보고서 등) |
| `chore` | 설정, 패키지 설치 등 자잘한 작업 |
| `refactor` | 기능 변화 없는 코드 정리 |
| `test` | 테스트 코드 관련 |

**예시**
```
feat: 가격 예측 함수 predict_price_range 구현
data: eBay CSV 결측치 처리 로직 추가
fix: Streamlit 폼 제출 시 오류 수정
docs: README 실행 방법 추가
```

### 브랜치 전략

- `main` : 항상 동작하는 상태만 유지 (바로 push 지양)
- `feature/{작업내용}` : 기능 단위로 브랜치 생성 후 작업

```bash
git checkout -b feature/price-prediction
git checkout -b feature/streamlit-ui
git checkout -b feature/agent-orchestration
```

작업 끝나면 `main`으로 Pull Request 생성 → 팀원 1명 이상 확인 후 merge

### 작업 전후 습관

```bash
# 작업 시작 전 최신 상태로 동기화
git pull origin main

# 작업 후
git add .
git commit -m "feat: 작업 내용"
git push origin feature/브랜치명
```

### 코드 스타일

- 함수/변수명: `snake_case`
- 커스텀 툴 함수는 역할이 드러나는 이름 사용 (예: `predict_price_range`)
- 커밋 전 `.env`가 스테이징에 포함되지 않았는지 `git status`로 확인
