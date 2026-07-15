import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 전처리 완료된 데이터 불러오기 (공유폴더 실행용 상대경로)
def load_processed_data(path: str = "iphone_resale_market_preprocessed.csv") -> pd.DataFrame:
    return pd.read_csv(path)

if __name__ == "__main__":
    df = load_processed_data()
    
    # 1. 결측치 처리 (학습에 쓸 핵심 컬럼들 중 빈칸 있는 행 전부 삭제)
    df = df.dropna(subset=[
        "price", "model_family", "generation_number", "is_pro", "storage_gb", "condition_score",
        "available", "sold", "us_state", "seller"
    ])

    # 2. 판매자(seller) 컬럼 인코딩 다이어트
    # 판매자 종류가 너무 많아서 그대로 원핫인코딩하면 컬럼 폭발함 (속도 저하, 오버핏 방지)
    # 매물 등록 수 기준 탑 15명만 살리고, 자잘한 개인 판매자들은 전부 'Others'로 묶음
    top_sellers = df["seller"].value_counts().nlargest(15).index
    df["seller"] = df["seller"].apply(lambda x: x if x in top_sellers else "Others")

    # 3. 타겟 변수(가격) 로그 변환
    # 중고 아이폰 특성상 가격 편차가 너무 심해서(우편향), log1p로 스케일 맞춰서 모델 안정성 확보
    y = np.log1p(df["price"])

    # 4. 학습용 독립변수(Feature) 선택
    X_raw = df[[
        "model_family", "generation_number", "is_pro", "storage_gb", "condition_score",
        "available", "sold", "us_state", "seller"
    ]]

    # 5. 원-핫 인코딩 (텍스트 데이터들 전부 0, 1 컬럼으로 변경)
    X = pd.get_dummies(
        X_raw, 
        columns=["model_family", "generation_number", "condition_score", "available", "sold", "us_state", "seller"],
        drop_first=True
    )

    # 6. Train / Test 데이터 분할 (8:2 비율, 재현성 위해 시드 고정)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. 오차 채점을 위해 로그 상태인 y_test 정답지를 원본 달러($) 가격으로 복원
    actual = np.expm1(y_test)

    # -------------------------------------------------------------
    # 랜덤포레스트 모델 학습 (GridSearchCV로 도출한 최적 파라미터 적용)
    # -------------------------------------------------------------
    print("RandomForestRegressor 모델 학습 진행 중...")
    
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5, 
        random_state=42, 
        n_jobs=-1 # 내 컴퓨터 CPU 코어 전부 다 쓰기
    )
    rf_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------
    # 예측 수행 및 모델 성능 채점
    # -------------------------------------------------------------
    # 혹시 모를 수치 폭발(inf) 막기 위해 클리핑 처리 후, expm1로 실제 달러 가격 복원
    preds_log = np.clip(rf_model.predict(X_test), 0, 15)
    preds = np.expm1(preds_log)
    
    # 원본 가격 기준으로 MAE(오차 금액)랑 R2 Score(설명력) 계산
    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(y_test, preds_log)

    # 보고서 첨부용 성능 요약 데이터프레임 만들기
    performance_df = pd.DataFrame([{
        "Model": "RandomForestRegressor",
        "MAE": f"${mae:.0f}",
        "R2(%)": f"{r2 * 100:.0f}%"
    }])
    
    # 터미널에 최종 결과 찍기
    print("\n" + "="*50)
    print("최종 모델 성능 결과 성적표")
    print("="*50)
    print(performance_df)

   
    # 학습 완료된 모델 자체를 pkl 파일로 저장
    joblib.dump(rf_model, "best_random_forest_model.pkl")
    print("\n[완료] 'best_random_forest_model.pkl' (최종 모델 객체) 저장 완료.")
    print("="*50 + "\n")