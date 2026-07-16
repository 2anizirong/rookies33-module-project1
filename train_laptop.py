# 라이브러리 로드
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 데이터셋 로드 및 확인
file_path = r'C:\module_project1\rookies33-module-project1\notebooks\0715ebay_laptops_model_ready_v1.csv'
df = pd.read_csv(file_path)

print(f"Dataset Shape: {df.shape}")
print(df.columns.tolist())
print(df.dtypes)
df.head()

# 독립변수(X)와 종속변수(y) 분리
X = df.drop(columns=["price_usd"])
y = df["price_usd"]
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 학습용 및 검증용 데이터 분할 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# 수치형 결측치 처리 (XGBoost 입력값 최적화)
num_cols = [
    'release_year', 'processor_speed_ghz', 'ram_gb', 'screen_size_inch', 
    'resolution_width', 'resolution_height', 'ssd_gb', 'hdd_gb', 
    'storage_capacity_gb', 'has_dual_storage', 'has_touchscreen', 
    'has_backlit_keyboard', 'has_bluetooth', 'has_webcam', 'has_wifi'
]

# 결측값 대입 시 SettingWithCopyWarning 방지를 위해 .loc 활용
X_train.loc[:, num_cols] = X_train[num_cols].replace(-1, np.nan)
X_test.loc[:, num_cols] = X_test[num_cols].replace(-1, np.nan)

# 범주형 데이터 원핫 인코딩
cat_cols = [
    'brand', 'model_family', 'cpu_brand', 'cpu_family', 
    'cpu_generation', 'cpu_suffix', 'os'
]

oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = oh_encoder.fit_transform(X_train[cat_cols])
X_test_cat = oh_encoder.transform(X_test[cat_cols])

# 수치형 데이터와 인코딩된 범주형 데이터 병합
X_train_final = np.hstack([X_train[num_cols].values, X_train_cat])
X_test_final = np.hstack([X_test[num_cols].values, X_test_cat])

# XGBoost 모델 정의 및 학습
xgb = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
xgb.fit(X_train_final, y_train)

# 학습용 및 검증용 예측치 생성
train_preds = xgb.predict(X_train_final)
test_preds = xgb.predict(X_test_final)

# 모델 예측 성능 평가
mae = mean_absolute_error(y_test, test_preds)
mse = mean_squared_error(y_test, test_preds)
rmse = np.sqrt(mse)
r2_train = r2_score(y_train, train_preds)
r2_test = r2_score(y_test, test_preds)

print("===== XGBoost Performance =====")
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2 (Train) : {r2_train:.3f}")
print(f"R2 (Test)  : {r2_test:.3f}")
print(f"R2 Gap     : {r2_train - r2_test:.3f}")

# 잔차(Residual) 분석 및 표준편차 계산
residuals = y_test - test_preds
residual_std = np.std(residuals)
print(f"Residual Std (오차 표준편차): {residual_std:.3f}")

# predict.py을 위한 변수명 리스트 추출
feature_columns = X_train.columns.tolist()

# 학습 완료된 모델 및 전처리 객체 저장
os.makedirs("models", exist_ok=True)
joblib.dump(oh_encoder, 'models/onehot_encoder.pkl')
joblib.dump({
    "model": xgb,
    "feature_columns": feature_columns,
    "residual_std": residual_std,
}, 'models/price_model_laptop.pkl')

print("인코더 저장 완료: models/onehot_encoder.pkl")
print("모델 파일 저장 완료: models/price_model_laptop.pkl")
