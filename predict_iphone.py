"""
예측 함수 (모델 학습 파트에서 관리, OpenAI Agent가 커스텀 툴로 호출)

역할:
- predict_price(): 학습된 머신러닝 모델로 직접 중고가 예측 (ML 모델 호출하기)
- detect_anomaly(): 예측가와 판매가를 비교해 저가/적정가/고가 판단
"""

import re

import joblib
import pandas as pd

CONDITION_SCORE_MAP = {
    "New": 7,
    "Excellent - Refurbished": 6,
    "Very Good - Refurbished": 5,
    "Good - Refurbished": 4,
    "Open Box": 3,
    "Used": 2,
    "For Parts Or Not Working": 1,
}

# 모델을 매번 파일에서 다시 불러오지 않도록 캐싱해두는 전역 변수
# (처음 한 번만 로드하고, 이후 호출부터는 메모리에 있는 걸 재사용)
# _이걸로 이름 앞에 붙이면 내부용이라는 관례적 신호. 외부(agent.py)에서 직접 건들이지 않기 
_model = None

# models/price_model.pkl 을 불러와서 반환하는 함수
# 파일 안에는 모델이랑 학습 시 사용한 컬럼 목록(feature_columns), 잔차 표준편차(residual_std: 모델이 평소에 얼마나 틀리는지)도 같이 딕셔너리 형태로 저장되어 있음
def _get_model():
    global _model
    if _model is None:
        # 처음 호출될 때만 실제로 디스크에서 파일을 읽어옴
        _model = joblib.load("models/price_model_iphone.pkl")
    return _model

# iphone 16 pro 이런 식으로 자유롭게 사용자에게서 입력 받은 텍스트 제품명에서 model_family / generation_number / is_pro 를 추출
# train.py 전처리 단계 -> train에서 전처리를 하나...? 데이터 전처리에서 하는 거 아닌가? 무튼 전처리 단계에서 이 값들을 실제로 어떻게 만들었는지와 같은 규칙이어야 함..
def _parse_title(title: str) -> dict:
    match = re.search(r"iPhone\s*(\d+)", title, re.IGNORECASE)

    generation_number = int(match.group(1)) if match else None
    is_pro = "pro" in title.lower()
    model_family = f"iPhone {generation_number}" if generation_number else title

    return {
        "model_family": model_family,
        "generation_number": generation_number,
        "is_pro": is_pro,
    }

# 사용자가 입력한 raw 값을 모델 학습 시 사용된 feature_columns에 맞춰 직접 매핑하는 함수
# refactor: get_dummies를 예측 시점 (아직 행이 1개)에 쓰면 안됨 -> 행이 1개면 범주형 칼럼의 고유값이 항상 1개라서 drop_first=True가 기준값으로 오인.. 칼럼 자체를 삭제함
# -> 모든 입력이 0으로 취급
# 학습 때 만들어진 feature_columns를 기준으로 수동 매핑하기 
def _prepare_input(features: dict, feature_columns: list) -> pd.DataFrame:
    df_input = pd.DataFrame(0, index=[0], columns=feature_columns)

    for key, value in features.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)   # True/False -> 1/0으로 변환
        if key in feature_columns:
            # 숫자형 컬럼(storage_gb, generation_number 등)은 컬럼명이 그대로 존재
            df_input.at[0, key] = value
        else:
            # 범주형 컬럼(model_family 등) -> "컬럼명_값" 형태로 매핑
            col_name = re.sub(r"[\[\]<>]", "_", f"{key}_{value}")
            if col_name in feature_columns:
                df_input.at[0, col_name] = 1
            # else: 학습 때 없던 값이면 전부 0으로 남음 (모델이 모르는 카테고리)

    return df_input

# ===========================================================================================
# 이 아래 두 함수는 OpenAI API Agent 팀이 직접 가져다 쓸 인터페이스라서 _를 안 붙였습니다
def predict_price(
    title: str,          # 제품명 (예: iPhone 16 Pro)
    storage_gb: int,     # 저장용량(GB)
    condition: str       # 제품 상태 등급 (tools 스키마의 enum 값과 일치해야 함)
) -> dict:
    """
    [ML 모델 호출] 입력받은 아이폰 정보를 ML 모델에 전달하여 적정 중고가를 예측한다.
    OpenAI Agent SDK의 function tool(predict_price)로 그대로 등록되는 함수

    Args:
        title: 제품명 (예: "iPhone 16 Pro")
        storage_gb: 저장용량(GB)
        condition: 제품 상태 등급 (tools 스키마의 enum 값과 일치해야 함)

    Returns:
        {
            "predicted_price": 450000,   # 모델이 예측한 적정가격(원)
            "residual_std": 32000.5,     # detect_anomaly()에서 이상 여부 판단 기준으로 사용
        }
    """

    artifact = _get_model()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    residual_std = artifact["residual_std"]

    raw_features = {
        **_parse_title(title),
        "storage_gb": storage_gb,
        "condition": condition,
    }

    X_input = _prepare_input(raw_features, feature_columns)

    # predict()는 배치 예측용이라 결과가 배열로 나옴 -> 입력이 1건이므로 [0]으로 첫 값만 추출
    predicted_price = model.predict(X_input)[0]

    return {
        "predicted_price": int(predicted_price),
        "residual_std": float(residual_std),
    }

def detect_anomaly(
    predicted_price: int,   # predict_price()가 반환한 모델의 적정가 예측값(원)
    selling_price: int,     # 사용자가 입력한 실제 판매 가격(원)
    residual_std: float     # predict_price()가 함께 반환한 모델 오차의 표준편차(원)
) -> dict:
    """
    [순수 Python 로직 - 모델 호출 없음]
    예측가격과 판매가격을 비교하여 저가/적정가/고가 여부를 판단한다.
    OpenAI Agent SDK의 function tool(detect_anomaly)로 그대로 등록되는 함수.

    Args:
        predicted_price: predict_price()가 반환한 모델의 적정가 예측값(원)
        selling_price: 사용자가 입력한 실제 판매 가격(원)
        residual_std: predict_price()가 함께 반환한 모델 오차의 표준편차(원)
    """

    # 방어 코드: 예측가 또는 판매가가 0 이하면 비교 자체가 의미 없으므로 조기 리턴
    if predicted_price <= 0 or selling_price <= 0:
        return {
            "status": "ERROR",
            "message": "예측 가격 또는 판매 가격이 유효하지 않아 비교할 수 없습니다.",
            "difference": None,
            "difference_percent": None,
        }
    
    # 판매가와 예측가의 차이 (양수면 판매가가 더 비쌈, 음수면 판매가가 더 쌈)
    difference = selling_price - predicted_price

    # 차이를 예측가 대비 퍼센트로 환산
    percent = (difference / predicted_price) * 100

    # 모델 오차 표준편차(residual_std) 대비 몇 배 벗어났는지
    deviation = abs(difference) / residual_std

    if deviation < 1:
        # 모델 오차 범위 이내 -> 정상적인 적정가 수준
        status = "적정가"
        message = "정상 거래 범위입니다."
    elif difference < 0:
        # 판매가가 예측가보다 낮은 경우
        if deviation < 2:
            status = "저가"
            message = "시세보다 다소 낮게 책정되어 있습니다."
        else:
            status = "저가"
            message = "시세보다 크게 낮아 허위 매물 가능성이 있으니 주의하세요."
    else:
        # 판매가가 예측가보다 높은 경우
        if deviation < 2:
            status = "고가"
            message = "시세보다 다소 높게 책정되어 있습니다."
        else:
            status = "고가"
            message = "시세보다 크게 높아 바가지 가능성이 있으니 주의하세요."

    return {
        "status": status,                        # "적정가" / "저가" / "고가" / "ERROR"
        "message": message,
        "difference": difference,                # 판매가 - 예측가 (원, 부호 있음)
        "difference_percent": round(percent, 2),
    }

if __name__ == "__main__":
    price_result = predict_price(title="iPhone 16 Pro", storage_gb=256, condition="Very Good - Refurbished")
    print("===== predict_price =====")
    print(price_result)

    anomaly_result = detect_anomaly(
        predicted_price=price_result["predicted_price"],
        selling_price=1200000,
        residual_std=price_result["residual_std"],
    )
    print("===== detect_anomaly =====")
    print(anomaly_result)