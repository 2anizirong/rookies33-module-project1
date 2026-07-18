# %% [markdown]
# 모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)
# 
# 역할:
# - data/processed/ 의 정제된 데이터로 적정가격 예측(회귀) 모델 학습
# - RandomForest / XGBoost / LinearRegression 성능 비교
# - 최종 모델 + 부가정보(feature_columns, residual_std)를 models/price_model.pkl 로 저장


# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 2. 데이터셋 로드 함수
def load_data(PATH):
    df = pd.read_csv(PATH)

    print(f'데이터 구성: {df.shape}')
    print(f"\n변수 타입 및 결측치 확인")
    print({df.info()})

    print(df['price_usd'].describe())

    numeric_df = df.select_dtypes(include=['number'])

    # 상관관계 히트맵 (Correlation Heatmap)
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap Matrix')
    plt.show()
    
    return df

# 3. 데이터 분리 함수
def split_data(df) :
    X = df.drop(columns=["price_usd"])
    y = df["price_usd"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42
    )
    return(X,X_train,X_test,y_train,y_test)


# 4. 데이터 전처리 함수
def preprocess_data(X,X_train,X_test):
    # 1. 숫자열 -1을 결측치로 변환
    num_cols = X.select_dtypes(include="number").columns.tolist()

    X_train[num_cols] = X_train[num_cols].replace(-1, np.nan)
    X_test[num_cols] = X_test[num_cols].replace(-1, np.nan)

    # 2. 문자열 데이터 OneHotEncoding
    cat_cols = X.select_dtypes(include="str").columns.tolist()
    oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train_cat = oh_encoder.fit_transform(X_train[cat_cols])
    X_test_cat = oh_encoder.transform(X_test[cat_cols])


    # 3. 수치형 데이터와 원핫 인코딩된 데이터 합치기
    X_train_final = np.hstack([X_train[num_cols].values, X_train_cat])
    X_test_final = np.hstack([X_test[num_cols].values, X_test_cat])

    # 4. predict.py에 넘겨줄 feature 변수 선언
    feature_columns = (num_cols + oh_encoder.get_feature_names_out(cat_cols).tolist())
    
    return (feature_columns,X_train_final, X_test_final)

# 5. 모델 학습 함수
def train_model(X_train,y_train):
    model = XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    return model

# 6. 성능 지표 계산 함수
def evaluate_model(model,X_train,X_test,y_train,y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_preds)
    mse = mean_squared_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2_train = r2_score(y_train,train_preds)
    r2_test = r2_score(y_test, test_preds)

    print("\n===== XGBoost =====")
    print(f"XGB_MAE: {mae}")
    print(f"XGB_MSE: {mse}")
    print(f"XGB_RMSE: {rmse}")
    print(f"XGB_R2: {r2_test:.3f}")
    print(f'XGB_R2_train: {r2_train:.3f}')
    print(f'XGB_Gap :{r2_train-r2_test:.3f}')
    
    residuals = y_test-test_preds
    residual_std = np.std(residuals)
    print(f"\nresidual_std: {residual_std}")
    
    return residual_std

# 7. 모델 저장 함수
def save_model(model,feature_columns,residual_std):
    # 현재 폴더에 models 폴더가 없으면 만듬
    os.makedirs("models", exist_ok=True)
    # 모델 저장
    joblib.dump(
    {
        "model" : model,
        "feature_columns" :feature_columns,
        "residual_std" : residual_std,
    },"models/price_model_laptop.pkl")

    print("저장완료 : models/price_model_laptop.pkl")

# 8. 실행 함수
def main():

    PATH = 'data/processed/0715ebay_laptops_model_ready_v1.csv'
    
    df = load_data(PATH)\
    
    X,X_train,X_test,y_train,y_test = split_data(df)

    feature_columns, X_train,X_test = preprocess_data(X,X_train,X_test)

    model = train_model(X_train,y_train)

    residual_std = evaluate_model(model,X_train,X_test,y_train,y_test)

    save_model(model,feature_columns,residual_std)

if __name__ == "__main__":
    main()