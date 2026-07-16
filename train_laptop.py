# %% [markdown]
# 모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)
# 
# 역할:
# - data/processed/ 의 정제된 데이터로 적정가격 예측(회귀) 모델 학습
# - RandomForest / XGBoost / LinearRegression 성능 비교
# - 최종 모델 + 부가정보(feature_columns, residual_std)를 models/price_model.pkl 로 저장

# %%
# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# %%
# 2. 데이터셋 로드
df = pd.read_csv('data/processed/0715ebay_laptops_model_ready_v1.csv')

# %%
X = df.drop(columns=["price_usd"])
y = df["price_usd"]


# %%
# 3. 데이터 분리(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# 3. 결측치 처리
num_cols = X.select_dtypes(include="number").columns.tolist()

X_train[num_cols] = X_train[num_cols].replace(-1, np.nan)
X_test[num_cols] = X_test[num_cols].replace(-1, np.nan)

# %%
# 4. 문자열 데이터 OneHotEncoding
cat_cols = X.select_dtypes(include="str").columns.tolist()
oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_train_cat = oh_encoder.fit_transform(X_train[cat_cols])
X_test_cat = oh_encoder.transform(X_test[cat_cols])

# %%
# 5. 수치형 데이터와 원핫 인코딩된 데이터 합치기
X_train_final = np.hstack([X_train[num_cols].values, X_train_cat])
X_test_final = np.hstack([X_test[num_cols].values, X_test_cat])

# %%
# 6. 모델 학습 및 예측 
xgb = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
xgb.fit(X_train_final, y_train)

train_preds = xgb.predict(X_train_final)
test_preds = xgb.predict(X_test_final)

# %%
# 7. 성능 지표 계산 및 출력
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

# %%
# 8. 가격 범위 계산에 쓸 residual 표준편차
residuals = y_test-test_preds
residual_std = np.std(residuals)
print(f"\nresidual_std: {residual_std}")

# %%
# 9. predict.py에서 인코딩을 똑같이 맞추기 위해 학습 때 쓴 컬럼 목록도 저장
feature_columns = oh_encoder.get_feature_names_out(cat_cols).tolist()

# %%
# 10. 모델 저장
# 현재 폴더에 models 폴더가 없으면 만듬
os.makedirs("models", exist_ok=True)

joblib.dump(oh_encoder, 'models/onehot_encoder.pkl')
joblib.dump(
{
    "model" : xgb,
    "feature_columns" :feature_columns,
    "residual_std" : residual_std,
},"models/price_model_laptop.pkl")

print('\n저장완료 : models/onehot_encoder.pkl')
print("저장완료 : models/price_model_laptop.pkl")