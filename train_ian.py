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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def main():
    # =====================================================
    # 1. 데이터 불러오기
    # =====================================================
    df = pd.read_csv("data/processed/iphone_resale_market_preprocessed.csv")

    print("데이터 크기:", df.shape)
    print(df.columns.tolist())

    # =====================================================
    # 2. Feature / Target 분리
    # =====================================================
    X = df.drop(columns=["price"])
    y = df["price"]

    # =====================================================
    # 3. One-Hot Encoding
    # =====================================================
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    print("범주형 컬럼:", cat_cols)

    X = pd.get_dummies(
        X,
        columns=cat_cols,
        drop_first=True
    )

    print("인코딩 후:", X.shape)

    # =====================================================
    # 4. Train / Test Split
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # =====================================================
    # 5. RandomForest
    # =====================================================
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)

    print("\n===== RandomForest =====")
    print(f"MAE  : {mean_absolute_error(y_test, rf_preds):.2f}")
    print(f"MSE  : {mean_squared_error(y_test, rf_preds):.2f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, rf_preds)):.2f}")
    print(f"R2   : {r2_score(y_test, rf_preds):.4f}")

    # =====================================================
    # 6. XGBoost
    # =====================================================
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()

    X_train_xgb.columns = [
        re.sub(r"[\[\]<>]", "_", c)
        for c in X_train_xgb.columns
    ]

    X_test_xgb.columns = [
        re.sub(r"[\[\]<>]", "_", c)
        for c in X_test_xgb.columns
    ]

    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(X_train_xgb, y_train)

    xgb_preds = xgb_model.predict(X_test_xgb)

    print("\n===== XGBoost =====")
    print(f"MAE  : {mean_absolute_error(y_test, xgb_preds):.2f}")
    print(f"MSE  : {mean_squared_error(y_test, xgb_preds):.2f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, xgb_preds)):.2f}")
    print(f"R2   : {r2_score(y_test, xgb_preds):.4f}")

    # =====================================================
    # 7. Linear Regression
    # =====================================================
    X_train_lr = X_train.fillna(X_train.mean())
    X_test_lr = X_test.fillna(X_train.mean())

    lr_model = LinearRegression()

    lr_model.fit(X_train_lr, y_train)

    lr_preds = lr_model.predict(X_test_lr)

    print("===== LinearRegression =====")
    print(f"MAE  : {mean_absolute_error(y_test, lr_preds):.2f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, lr_preds)):.2f}")
    print(f"R2   : {r2_score(y_test, lr_preds):.4f}")

    # =====================================================
    # 8. 모델 저장
    # =====================================================
    final_model = rf_model
    final_preds = rf_preds

    residual_std = np.std(y_test - final_preds)

    feature_columns = X_train.columns.tolist()

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "residual_std": residual_std,
        },
        "models/price_model.pkl",
    )

    print("모델 저장 완료!")
    print("models/price_model.pkl")


if __name__ == "__main__":
    main()