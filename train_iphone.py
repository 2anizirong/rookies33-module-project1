"""
모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)

역할:
- data/processed/ 의 정제된 데이터로 아이폰 중고가격 예측 모델 학습
- RandomForest / XGBoost / LinearRegression 성능 비교
- 가장 성능이 좋은 모델 선택
- 최종 모델 + 예측에 필요한 정보(feature_columns, residual_std) 저장

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
    """

    try:
        df = pd.read_csv(file_path)

        print(f'데이터 구성 : {df.shape}')

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
        print('파일 로드 중 에러가 발생했습니다.')
        return None



# 3. 데이터 분리 함수
def split_data(df):

    # 데이터 가격 단위 변환
    # 원본 데이터는 달러($) 기준이므로 서비스 입력 기준인 원화(₩)로 변환
    USD_TO_KRW = 1400


    # feature(X) 와 target(y) 분리
    X = df.drop(columns=["price"])

    # 모델이 예측할 가격
    # 달러 → 원화 변환 후 학습
    y = df["price"] * USD_TO_KRW



    # train 데이터와 test 데이터 분리
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

    # 결측치 처리
    # 범주형 데이터(color) → 최빈값으로 대체

    if "color" in X_train.columns:

        mode_color = X_train["color"].mode()[0]

        X_train["color"] = X_train["color"].fillna(mode_color)

        X_test["color"] = X_test["color"].fillna(mode_color)



    # 숫자형 데이터 결측치 → 중앙값으로 대체

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



    # predict.py에서 입력하는 데이터 타입과 동일하게 변환

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



    # 문자열 데이터 확인
    # 모델 학습을 위해 One-Hot Encoding 적용

    cat_cols = X.select_dtypes(
        include="object"
    ).columns.tolist()


    print("범주형 컬럼:", cat_cols)



    # 범주형 데이터를 숫자 컬럼으로 변환

    X_train = pd.get_dummies(
        X_train,
        columns=cat_cols,
        drop_first=True
    )


    X_test = pd.get_dummies(
        X_test,
        columns=cat_cols,
        drop_first=True
    )



    # Test 데이터 컬럼을 Train 데이터 기준으로 맞춤
    # 학습 시 존재하지 않는 컬럼은 0으로 생성

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )



    # 예측 시 입력 데이터를 동일한 구조로 만들기 위해 저장

    feature_columns = X_train.columns.tolist()


    return (
        feature_columns,
        X_train,
        X_test
    )



# 5. 모델 학습 함수

def train_model(X_train, y_train):


    # RandomForest 모델 생성

    base_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=9,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )


    # Target 데이터를 log 변환 후 학습
    # 가격 데이터의 큰 편차를 줄이고 예측 안정성을 높이기 위해 사용
    # 예측 후 자동으로 원래 가격 단위로 복원

    rf_model = TransformedTargetRegressor(
        regressor=base_rf,
        func=np.log1p,
        inverse_func=np.expm1
    )


    rf_model.fit(
        X_train,
        y_train
    )



    # XGBoost 모델 생성

    base_xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )


    xgb_model = TransformedTargetRegressor(
        regressor=base_xgb,
        func=np.log1p,
        inverse_func=np.expm1
    )


    xgb_model.fit(
        X_train,
        y_train
    )



    # Linear Regression 모델 생성

    base_lr = LinearRegression()


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


    models = {

        "RandomForestRegressor": rf_model,

        "XGBoostRegressor": xgb_model,

        "LinearRegressor": lr_model

    }


    results = {}



    for name, model in models.items():


        # train / test 데이터 예측

        train_preds = model.predict(X_train)

        test_preds = model.predict(X_test)



        # R2 계산

        train_r2 = r2_score(
            y_train,
            train_preds
        )

        test_r2 = r2_score(
            y_test,
            test_preds
        )



        # 오차 계산

        mae = mean_absolute_error(
            y_test,
            test_preds
        )

        mse = mean_squared_error(
            y_test,
            test_preds
        )

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

        print(f"R2_Gap : {train_r2-test_r2:.3f}")

        print(f"MAE: {mae}")

        print(f"RMSE: {rmse}")



    # Test R2 점수가 가장 높은 모델 선택

    best_model_name = max(
        results,
        key=lambda x: results[x]["test_r2"]
    )


    best_model = results[best_model_name]["model"]



    # 모델 예측 오차 계산
    # detect_anomaly()에서 이상 여부 판단 기준으로 사용

    best_preds = best_model.predict(X_test)

    residual = y_test - best_preds


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


    os.makedirs(
        "models",
        exist_ok=True
    )


    # 모델과 예측에 필요한 정보를 함께 저장

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

    file_path = (
        "data/processed/"
        "iphone_clean_processed.csv"
    )


    df = load_data(file_path)


    X, X_train, X_test, y_train, y_test = split_data(df)


    feature_columns, X_train, X_test = preprocess_data(
        X,
        X_train,
        X_test
    )


    rf_model, xgb_model, lr_model = train_model(
        X_train,
        y_train
    )


    best_model, residual_std = evaluate_model(
        rf_model,
        xgb_model,
        lr_model,
        X_train,
        X_test,
        y_train,
        y_test
    )


    save_model(
        best_model,
        feature_columns,
        residual_std
    )



if __name__ == "__main__":
    main()