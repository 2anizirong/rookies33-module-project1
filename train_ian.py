"""
모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)

역할:
- data/processed/ 의 정제된 데이터로 적정가격 예측(회귀) 모델 학습
- RandomForest / XGBoost / LinearRegression 성능 비교
- 최종 모델 + 부가정보(feature_columns, residual_std)를 models/price_model.pkl 로 저장
"""

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 데이터 경로 / target 컬럼명 (데이터 바뀌면 여기만 수정하면 됨)
DATA_PATH = "data/processed/iphone_resale_market_preprocessed.csv"
TARGET_COL = "price"


def load_processed_data(path: str = DATA_PATH) -> pd.DataFrame:
    """전처리 완료된 CSV 로드"""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    학습용 X, y 분리 + 범주형 컬럼 원-핫 인코딩 + XGBoost용 컬럼명 정리

    Returns:
        X, y (인코딩 및 컬럼명 정리까지 끝난 상태)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    print("범주형 컬럼:", cat_cols)
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X.columns = [re.sub(r"[\[\]<>]", "_", col) for col in X.columns]

    return X, y


def train_model(X_train, y_train):
    """RandomForest 기준 베이스라인 모델 학습"""
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, name: str = "Model") -> dict:
    """MAE / RMSE / R2 계산 및 출력"""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"===== {name} =====")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.3f}")

    return {"preds": preds, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def compare_models(X_train, X_test, y_train, y_test) -> dict:
    """
    RandomForest / XGBoost / LinearRegression 세 모델을 학습하고 성능 비교.
    LinearRegression은 결측치가 있으면 에러가 나므로 결측치를 채운 데이터를 별도로 사용.
    """
    results = {}

    rf_model = train_model(X_train, y_train)
    rf_result = evaluate_model(rf_model, X_test, y_test, name="RandomForest")
    rf_result["model"] = rf_model
    results["RandomForest"] = rf_result

    xgb_model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_result = evaluate_model(xgb_model, X_test, y_test, name="XGBoost")
    xgb_result["model"] = xgb_model
    results["XGBoost"] = xgb_result

    X_train_filled = X_train.fillna(X_train.mean())
    X_test_filled = X_test.fillna(X_train.mean())

    lr_model = LinearRegression()
    lr_model.fit(X_train_filled, y_train)
    lr_result = evaluate_model(lr_model, X_test_filled, y_test, name="LinearRegression")
    lr_result["model"] = lr_model
    results["LinearRegression"] = lr_result

    return results


def save_model_artifact(model, feature_columns: list, residual_std: float, path: str = "models/price_model.pkl"):
    """최종 모델 + 부가정보(feature_columns, residual_std)를 하나로 묶어서 저장"""
    joblib.dump(
        {"model": model, "feature_columns": feature_columns, "residual_std": residual_std},
        path,
    )
    print(f"모델 저장 완료: {path}")


if __name__ == "__main__":
    df = load_processed_data()
    print(df.shape)

    X, y = prepare_features(df)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)

    missing = X_train.isnull().sum()
    print(missing[missing > 0])

    results = compare_models(X_train, X_test, y_train, y_test)

    # ===== (임시) 최종 모델 선택 =====
    FINAL_MODEL_NAME = "RandomForest"

    final_result = results[FINAL_MODEL_NAME]
    final_model = final_result["model"]
    final_preds = final_result["preds"]

    residual_std = np.std(y_test - final_preds)
    print(f"residual_std: {residual_std:.2f}")

    feature_columns = X_train.columns.tolist()

    save_model_artifact(final_model, feature_columns, residual_std)