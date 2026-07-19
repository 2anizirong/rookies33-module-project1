"""
모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)

역할:
- data/processed/ 의 정제된 데이터로 아이폰 중고가격 예측 모델 학습
- RandomForest / XGBoost / LinearRegression 성능 비교
- Test 성능 기준 최종 모델 선택
- 예측에 필요한 정보(feature_columns, residual_std)와 함께 모델 저장

저장 파일:
models/price_model_iphone.pkl
"""


# 1. 라이브러리 불러오기

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor



# 2. 데이터 로드 함수

def load_data(file_path):
    """
    CSV 데이터를 불러오는 함수

    확인 내용:
    - 데이터 크기
    - 컬럼 목록
    - 데이터 타입
    - 샘플 데이터
    """

    try:
        df = pd.read_csv(file_path)

        print(f"데이터 구성 : {df.shape}")

        print("\n컬럼 목록")
        print(df.columns.tolist())

        print("\n데이터 타입")
        print(df.dtypes)

        print("\n상위 데이터")
        print(df.head())

        return df

    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return None

    except Exception as e:
        print("파일 로드 중 에러가 발생했습니다.")
        return None



# 3. 데이터 분리 함수

def split_data(df):

    # 데이터셋의 price는 달러($) 기준이므로
    # 서비스에서 사용하는 원화(₩) 기준으로 변환
    USD_TO_KRW = 1400

    # Feature(X)와 Target(y) 분리
    # price를 제외한 나머지 컬럼은 입력 데이터
    X = df.drop(columns=["price"])

    # 예측 대상 가격
    # 달러 가격 -> 원화 가격 변환 후 학습
    y = df["price"] * USD_TO_KRW

    # Train/Test 데이터 분리
    # Train : 모델 학습
    # Test : 학습하지 않은 데이터 성능 평가

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X,
        X_train,
        X_test,
        y_train,
        y_test
    )



# 4. 데이터 전처리 함수

def preprocess_data(X, X_train, X_test):

    # ------------------------------
    # 결측치 처리
    # ------------------------------

    # 범주형 데이터(color)는 최빈값으로 처리

    if "color" in X_train.columns:

        mode_color = X_train["color"].mode()[0]

        X_train["color"] = X_train["color"].fillna(mode_color)
        X_test["color"] = X_test["color"].fillna(mode_color)


    # 숫자형 데이터(battery_health)는 중앙값으로 처리

    if "battery_health" in X_train.columns:

        median_battery = X_train["battery_health"].median()

        X_train["battery_health"] = (
            X_train["battery_health"]
            .fillna(median_battery)
        )

        X_test["battery_health"] = (
            X_test["battery_health"]
            .fillna(median_battery)
        )


    # ------------------------------
    # 데이터 타입 변환
    # ------------------------------

    # predict.py에서 입력하는 값과 동일한 타입으로 변환

    numeric_cols = [
        "generation_number",
        "is_pro",
        "storage_gb",
        "condition_score"
    ]

    for col in numeric_cols:

        if col in X_train.columns:

            X_train[col] = pd.to_numeric(
                X_train[col],
                errors="coerce"
            )

            X_test[col] = pd.to_numeric(
                X_test[col],
                errors="coerce"
            )


    # ------------------------------
    # One-Hot Encoding
    # ------------------------------

    # 문자열 데이터를 숫자 형태로 변환
    # 머신러닝 모델은 문자열을 직접 처리할 수 없음

    cat_cols = X.select_dtypes(
        include="object"
    ).columns.tolist()

    print("범주형 컬럼:", cat_cols)


    # Train 데이터 기준 One-Hot Encoding

    X_train = pd.get_dummies(
        X_train,
        columns=cat_cols,
        drop_first=True
    )


    # Test 데이터도 동일한 방식 적용

    X_test = pd.get_dummies(
        X_test,
        columns=cat_cols,
        drop_first=True
    )


    # Train과 Test의 컬럼 불일치 방지
    # 학습 때 존재하지 않는 컬럼은 0으로 추가

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )


    # predict.py에서 동일한 입력 구조를 만들기 위해 저장
    feature_columns = X_train.columns.tolist()


    return (
        feature_columns,
        X_train,
        X_test
    )

# 5. 모델 학습 함수

def train_model(X_train, y_train):

    # ==============================
    # RandomForest 모델
    # ==============================

    base_rf = RandomForestRegressor(
        n_estimators=200,       # 생성할 결정 트리 개수
        max_depth=10,            # 트리 최대 깊이 제한
        min_samples_leaf=3,     # leaf 노드 최소 샘플 수
        random_state=42,
        n_jobs=-1               # 모든 CPU 코어 사용
    )


    # Target 로그 변환 적용
    # 가격 데이터는 고가/저가 차이가 크기 때문에
    # 로그 변환 후 학습하면 극단값 영향을 줄일 수 있음
    #
    # 학습: log1p(y)
    # 예측: expm1()을 통해 원래 가격 단위로 자동 복원

    rf_model = TransformedTargetRegressor(
        regressor=base_rf,
        func=np.log1p,
        inverse_func=np.expm1
    )


    rf_model.fit(
        X_train,
        y_train
    )



    # ==============================
    # XGBoost 모델
    # ==============================

    base_xgb = XGBRegressor(
        n_estimators=200,       # 부스팅 반복 횟수
        # max_depth=6,            # 트리 깊이
        # learning_rate=0.1,      # 학습률
        random_state=42,
        n_jobs=-1
    )


    # XGBoost도 동일하게 Target 로그 변환 적용

    xgb_model = TransformedTargetRegressor(
        regressor=base_xgb,
        func=np.log1p,
        inverse_func=np.expm1
    )


    xgb_model.fit(
        X_train,
        y_train
    )



    # ==============================
    # Linear Regression 모델
    # ==============================

    base_lr = LinearRegression()


    # Linear Regression도 동일한 Target 변환 적용

    lr_model = TransformedTargetRegressor(
        regressor=base_lr,
        func=np.log1p,
        inverse_func=np.expm1
    )


    lr_model.fit(
        X_train,
        y_train
    )


    return (
        rf_model,
        xgb_model,
        lr_model
    )



# 6. 모델 성능 평가 함수

def evaluate_model(
        rf_model,
        xgb_model,
        lr_model,
        X_train,
        X_test,
        y_train,
        y_test
    ):


    # 비교할 모델 저장

    models = {
        "RandomForestRegressor": rf_model,
        "XGBoostRegressor": xgb_model,
        "LinearRegressor": lr_model
    }


    results = {}


    # 각각의 모델 성능 비교

    for name, model in models.items():

        # Train/Test 데이터 예측

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)


        # R2 Score 계산
        # 1에 가까울수록 예측력이 좋음

        train_r2 = r2_score(
            y_train,
            train_preds
        )

        test_r2 = r2_score(
            y_test,
            test_preds
        )


        # 평균 절대 오차

        mae = mean_absolute_error(
            y_test,
            test_preds
        )


        # 평균 제곱 오차

        mse = mean_squared_error(
            y_test,
            test_preds
        )


        # RMSE
        # 실제 가격 단위와 동일하여 해석하기 쉬움

        rmse = np.sqrt(mse)



        results[name] = {
            "model": model,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        }



        print(f"===== {name} =====")

        print(f"R2_train: {train_r2:.3f}")

        print(f"R2_test: {test_r2:.3f}")

        # Train/Test 성능 차이
        # 값이 크면 과적합 가능성 확인

        print(f"R2_Gap : {train_r2-test_r2:.3f}")

        print(f"MAE: {mae}")

        print(f"RMSE: {rmse}")



    # Test R2 점수가 가장 높은 모델 선택

    best_model_name = max(
        results,
        key=lambda x: results[x]["test_r2"]
    )


    best_model = results[best_model_name]["model"]



    # 선택된 모델의 예측 오차 계산

    best_preds = best_model.predict(X_test)


    residual = y_test - best_preds


    # 모델이 평균적으로 얼마나 틀리는지 나타내는 값
    # detect_anomaly()에서 가격 이상 판단 기준으로 사용

    residual_std = np.std(residual)



    print(f"Best Model : {best_model_name}")

    print(f"Residual STD : {residual_std:.3f}")



    return (
        best_model,
        residual_std
    )



# 7. 모델 저장 함수

def save_model(
        model,
        feature_columns,
        residual_std
    ):


    # models 폴더가 없으면 생성

    os.makedirs(
        "models",
        exist_ok=True
    )


    # 모델과 예측에 필요한 정보 저장
    #
    # model:
    # 실제 학습된 머신러닝 모델
    #
    # feature_columns:
    # 학습 당시 사용한 입력 컬럼 목록
    # -> predict.py에서 동일한 구조 생성 시 사용
    #
    # residual_std:
    # 모델 평균 오차 범위
    # -> 가격 이상 탐지 기준으로 사용

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "residual_std": residual_std
        },

        "models/price_model_iphone.pkl"
    )


    print("\n저장 완료: models/price_model_iphone.pkl")

    # 8. 실행 함수

def main():

    # 학습 데이터 경로 지정

    file_path = "data/processed/iphone_clean_processed.csv"
    
    # 데이터 로드

    df = load_data(file_path)


    # Feature(X)와 Target(y) 분리
    # Train/Test 데이터 생성

    X, X_train, X_test, y_train, y_test = split_data(df)



    # 데이터 전처리

    # 반환값:
    # feature_columns:
    #   - 학습에 사용된 최종 컬럼 목록
    #   - predict.py에서 동일한 입력 구조 생성 시 사용
    #
    # X_train:
    #   - 전처리 완료된 학습 데이터
    #
    # X_test:
    #   - 전처리 완료된 테스트 데이터

    feature_columns, X_train, X_test = preprocess_data(
        X,
        X_train,
        X_test
    )



    # RandomForest / XGBoost / LinearRegression 학습

    rf_model, xgb_model, lr_model = train_model(
        X_train,
        y_train
    )



    # 세 모델 성능 비교 후
    # Test R2가 가장 높은 모델 선택

    best_model, residual_std = evaluate_model(
        rf_model,
        xgb_model,
        lr_model,
        X_train,
        X_test,
        y_train,
        y_test
    )



    # 최종 선택 모델 저장

    save_model(
        best_model,
        feature_columns,
        residual_std
    )



# 해당 파일을 직접 실행했을 때만 학습 수행

if __name__ == "__main__":
    main()