# %% [markdown]
# 모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)
#
# 역할:
# - data/processed/ 의 정제된 데이터로 노트북 중고가격 예측(회귀) 모델 학습
# - XGBoost Regression 모델 학습
# - 모델 성능 평가 및 예측 오차 계산
# - 최종 모델 + 부가정보(feature_columns, residual_std)를 저장
#
# 저장 파일:
# models/price_model_laptop.pkl


# =========================
# 1. 라이브러리 불러오기
# =========================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import joblib



# =========================
# 2. 데이터셋 로드 함수
# =========================

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



# =========================
# 3. 데이터 분리 함수
# =========================

def split_data(df):

    """
    Feature(X)와 Target(y)를 분리하고
    학습 데이터와 테스트 데이터로 분리하는 함수
    """


    # 가격 컬럼 제외한 나머지 컬럼을 Feature로 사용

    X = df.drop(
        columns=["price_usd"]
    )


    # 예측 대상 가격

    y = df["price_usd"]



    # 학습 데이터 / 테스트 데이터 분리

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



# =========================
# 4. 데이터 전처리 함수
# =========================

def preprocess_data(
        X,
        X_train,
        X_test
    ):


    # -------------------------
    # 숫자형 결측치 처리
    # -------------------------

    # 데이터에서 -1은 결측값 의미로 사용되므로 NaN으로 변환

    num_cols = X.select_dtypes(
        include="number"
    ).columns.tolist()



    X_train[num_cols] = (
        X_train[num_cols]
        .replace(-1, np.nan)
    )


    X_test[num_cols] = (
        X_test[num_cols]
        .replace(-1, np.nan)
    )



    # ----------------------------
    # 범주형 데이터 One-Hot Encoding
    # ----------------------------

    # 문자열 컬럼 추출

    cat_cols = X.select_dtypes(
        include="str"
    ).columns.tolist()



    # 학습 데이터 기준으로 Encoder 생성
    # handle_unknown='ignore'
    # → 학습하지 않은 새로운 카테고리가 입력되어도 오류 방지

    oh_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )



    X_train_cat = oh_encoder.fit_transform(
        X_train[cat_cols]
    )


    X_test_cat = oh_encoder.transform(
        X_test[cat_cols]
    )



    # -------------------------
    # 숫자형 + One-Hot 데이터 결합
    # -------------------------

    X_train_final = np.hstack(
        [
            X_train[num_cols].values,
            X_train_cat
        ]
    )


    X_test_final = np.hstack(
        [
            X_test[num_cols].values,
            X_test_cat
        ]
    )



    # predict.py에서 입력 데이터 변환 시
    # 학습 당시 컬럼 구조를 맞추기 위해 저장

    feature_columns = (
        num_cols
        +
        oh_encoder
        .get_feature_names_out(cat_cols)
        .tolist()
    )


    return (
        feature_columns,
        X_train_final,
        X_test_final
    )



# =========================
# 5. 모델 학습 함수
# =========================

def train_model(
        X_train,
        y_train
    ):


    # XGBoost Regression 모델 생성

    model = XGBRegressor(

        random_state=42,

        n_estimators=100,

        learning_rate=0.1

    )


    # 모델 학습

    model.fit(
        X_train,
        y_train
    )


    return model



# =========================
# 6. 모델 성능 평가 함수
# =========================

def evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ):


    # train / test 데이터 예측

    train_preds = model.predict(X_train)

    test_preds = model.predict(X_test)



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



    # 모델 설명력 계산

    r2_train = r2_score(
        y_train,
        train_preds
    )


    r2_test = r2_score(
        y_test,
        test_preds
    )



    print("\n===== XGBoost =====")

    print(f"XGB_MAE : {mae}")

    print(f"XGB_MSE : {mse}")

    print(f"XGB_RMSE : {rmse}")

    print(f"XGB_R2_test : {r2_test:.3f}")

    print(f"XGB_R2_train : {r2_train:.3f}")

    print(f"XGB_Gap : {r2_train-r2_test:.3f}")



    # 실제 가격과 예측 가격 차이 계산

    residuals = y_test - test_preds



    # 모델 평균 오차
    # detect_anomaly()에서 이상 여부 판단 기준으로 사용

    residual_std = np.std(residuals)



    print(
        f"\nresidual_std : {residual_std}"
    )


    return residual_std



# =========================
# 7. 모델 저장 함수
# =========================

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



    # 모델과 예측에 필요한 정보를 함께 저장

    joblib.dump(

        {
            "model": model,

            "feature_columns": feature_columns,

            "residual_std": residual_std
        },

        "models/price_model_laptop.pkl"

    )


    print(
        "저장완료 : models/price_model_laptop.pkl"
    )



# =========================
# 8. 실행 함수
# =========================

def main():


    file_path = (
        "data/processed/0715ebay_laptops_model_ready_v1.csv"
    )



    # 데이터 로드

    df = load_data(file_path)



    # 데이터 분리

    X, X_train, X_test, y_train, y_test = split_data(df)



    # 데이터 전처리

    feature_columns, X_train, X_test = preprocess_data(
        X,
        X_train,
        X_test
    )



    # 모델 학습

    model = train_model(
        X_train,
        y_train
    )



    # 모델 평가 및 residual_std 계산

    residual_std = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )



    # 모델 저장

    save_model(
        model,
        feature_columns,
        residual_std
    )



if __name__ == "__main__":

    main()