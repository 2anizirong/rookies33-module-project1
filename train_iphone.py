"""
모델 학습 담당 (모델 학습 파트: 김윤호, 김이안)

역할:
- data/processed/ 의 정제된 데이터로 적정가격 예측(회귀) 모델 학습
- RandomForest / XGBoost / LinearRegression 성능 비교
- 최종 모델 + 부가정보(feature_columns, residual_std)를 models/price_model.pkl 로 저장
"""

# %% [markdown]
# 라이브러리 불러오기

# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import re
from sklearn.linear_model import LinearRegression

# %% [markdown]
# RandomForest

# %%
df = pd.read_csv('data/processed/iphone_clean_processed.csv')
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
df.head()


# numeric_df = df.select_dtypes(include=['number'])

# # 상관관계 히트맵 (Correlation Heatmap)
# plt.figure(figsize=(10, 8))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Correlation Heatmap Matrix')
# plt.show()

# %% [markdown]
# Train/Test 분리 + 전처리 확인

# %%
# [c for c in df.columns if "price" in c.lower()]

# %%
# price_cols = [c for c in df.columns if "price" in c.lower()]
X = df.drop(columns=["price"])
y = df["price"]

print(X.shape, y.shape)
X.head()

# %%
# 범주형 데이터 원핫인코딩 필요
cat_cols = X.select_dtypes(include="object").columns.tolist()
print("범주형 컬럼:", cat_cols)
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 모든 모델 공통 컬럼명 정리
X.columns = [
    re.sub(r"[\[\]<>]", "_", col)
    for col in X.columns
]

print(X.shape)
X.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# %%
y_train_log = np.log1p(y_train)

# %%
# train_test_split 직후, 모델 학습 전에 한번 확인
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

# %% [markdown]
# RandomForestRegressor 학습 + 평가 지표

# %%
# rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1) #: 1차 모델학습 파라미터

base_rf = RandomForestRegressor(
    n_estimators=200,        
    max_depth=9,          
    min_samples_leaf=1,              
    random_state=42,
    n_jobs=-1
)
rf_model = TransformedTargetRegressor( # 로그함수 자동 역변환
    regressor=base_rf,
    func=np.log1p,
    inverse_func=np.expm1
)



rf_model.fit(X_train, y_train)

train_preds = rf_model.predict(X_train)
test_preds = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, test_preds)
mse = mean_squared_error(y_test, test_preds)
rmse = np.sqrt(mean_squared_error(y_test, test_preds))
r2_train = r2_score(y_train,train_preds)
r2_test = r2_score(y_test, test_preds)

print("===== RandomForest =====")
print(f"RF_MAE: {mae}")
print(f"RF_MSE: {mse}")
print(f"RF_RMSE: {rmse}")
print(f"RF_R2: {r2_test:.3f}")
print(f'RF_R2_train: {r2_train:.3f}')
print(f'R2_Gap :{r2_train-r2_test:.3f}')



# %% [markdown]
# ### 평가 기준
# 
# 과적합 판단
# - train r2와 test r2의 차이가 0.2 이상이면  과적합으로 판단
# 
# 
# 도메인별 현실적인 R² 스코어 기준
# - 0.8 ~ 0.9 이상 (최상위권)
# - 0.5 ~ 0.7 (우수함)
# - 0.2 ~ 0.4 (수용 가능)
# 
# 
# Train_R2_Score 기준 과소적합 판단
# - -0.05 이내 하락 (미세 조정)
# - -0.05 ~ -0.15 하락 (강한 규제)
# - -0.15 이상 폭락 (과소적합 발생)

# %% [markdown]
# ### 하이퍼파라미터 튜닝 실험결과
# 
# 실험 1차 결과 => 기본
# RF_MAE: 121.43745249595051
# RF_MSE: 131474.1940341905
# RF_RMSE: 362.5937037983292
# RF_R2: 0.429
# RF_R2_train: 0.539
# R2_Gap :0.110
# R2 값이 너무 낮음 신뢰성이 없음
# 
# 
# 실험 2차 결과 => max_depth = 10 추가 
# - RF_R2: 0.535
# - RF_R2_train: 0.736 
# - R2_Gap :0.201
# - 상태 : 과적합 경계선
# - 방향성 : 말단노드에 남겨야 할 최소 데이터의 수를 제한(남겨야할 데이터가 많으면 학습을 포기할수 있음)
# 
# 
# 실험 3차 결과 => min_samples_leaf=2 추가
# - RF_R2: 0.523
# - RF_R2_train: 0.584 (훈련점수 갭 : 0.152)
# - R2_Gap :0.061
# - 상태: 과소적합 => 2차와의 Train_R2_Score Gap이 0.152로 과소적합 발생
# - 방향성 => 과소적합 발생
# 
# 
# 실험 4차 결과 => max_depth = 11 변경
# - RF_R2: 0.525
# - RF_R2_train: 0.594 (훈련점수 갭 : 0.142) -> 강한 규제 상태로 보임
# - R2_Gap :0.069
# - 방향성 => min_samples_leaf=1 (R2_Score를 올리기 위해 규제를 풀어봄)
# 
# 
# 실험 5차 결과 => min_samples_leaf=1(규제를 풀음)
# - RF_R2: 0.537
# - RF_R2_train: 0.758 
# - R2_Gap :0.221
# - 과적합 
# - 방향성 : RF_R2 값은 올라갔지만 규제를 풀자마자 과적합 발생, max_depth = 9로 변경
# 
# 실험 6차 결과 => max_depth = 9 변경
# - RF_R2: 0.532 (우수함)
# - RF_R2_train: 0.712(훈련점수 갭 : -0.046)
# - R2_Gap :0.181 (과적합 아님)
# - 일반화 상태
# - 최종 튜닝 파라미터 : max_depth = 9, min_samples_leaf=1
# 

# %% [markdown]
# ----
# XGBoost

# %%
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
)

# ========================================================
# XGBoost가 허용하지 않는 특수문자를 언더스코어로 치환
X_train.columns = [re.sub(r"[\[\]<>]", "_", col) for col in X_train.columns]
X_test.columns = [re.sub(r"[\[\]<>]", "_", col) for col in X_test.columns]
# =========================================================

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_train_preds = xgb_model.predict(X_train)

xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_mse = mean_squared_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)

print("\n===== XGBoost =====")
print(f"XGB_MAE: {xgb_mae}")
print(f"XGB_MSE: {xgb_mse}")
print(f"XGB_RMSE: {xgb_rmse}")
print(f"XGB_R2: {xgb_r2}")

# %% [markdown]
# ---
# LinearRegression

# %%
X_train.isnull().sum()[X_train.isnull().sum() > 0]

# %%
# 학습 데이터 기준으로 결측치 채우기
X_train_filled = X_train.fillna(X_train.mean())
X_test_filled = X_test.fillna(X_train.mean())  # test도 train의 평균값 기준으로 채워야 함 (data leakage 방지)

# %%
lr_model = LinearRegression()
lr_model.fit(X_train_filled, y_train)

lr_preds = lr_model.predict(X_test_filled)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)

print("\n===== LinearRegression =====")
print(f"LR_MAE: {lr_mae}")
print(f"LR_RMSE: {lr_rmse}")
print(f"LR_R2: {lr_r2}")

# %%
# 확인해보기


# %%

# %%
# 최종 모델 선택 (여기선 RandomForest 예시)
final_model = rf_model
final_preds = test_preds  # rf_model의 predict 결과

# 가격 범위 계산에 쓸 residual 표준편차
residuals = y_test - final_preds
residual_std = np.std(residuals)
print(f"residual_std: {residual_std}")

# predict.py에서 인코딩을 똑같이 맞추기 위해 학습 때 쓴 컬럼 목록도 저장
feature_columns = X_train.columns.tolist()

# %%
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": final_model,
    "feature_columns": feature_columns,
    "residual_std": residual_std,
}, "models/price_model_iphone.pkl")

print("\n저장 완료: models/price_model_iphone.pkl")