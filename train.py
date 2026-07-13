"""
모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)

역할:
- data/processed/ 의 정제된 데이터로 적정가격 예측(회귀) 모델 학습
- RandomForest / XGBoost 등 후보 모델 비교
- 최종 모델을 models/price_model.pkl 로 저장
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def load_processed_data(path: str = "data/processed/laptops_clean.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(X_train, y_train):
    """RandomForest 기준 베이스라인. XGBoost 등과 성능 비교 후 최종 모델 선택"""
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.2f} / R2: {r2:.3f}")
    return {"mae": mae, "r2": r2}


if __name__ == "__main__":
    df = load_processed_data()

    # TODO: 실제 feature/target 컬럼명에 맞게 수정
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, "models/price_model.pkl")
    print("모델 저장 완료: models/price_model.pkl")
