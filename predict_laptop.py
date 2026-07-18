"""
예측 함수 (모델 학습 파트에서 관리, OpenAI Agent가 커스텀 툴로 호출)
 
역할:
- predict_price_laptop(): 학습된 머신러닝 모델로 중고 노트북 적정가 예측 (ML 모델 호출)
- detect_anomaly(): 예측가와 판매가를 비교해 저가/적정가/고가 판단
 
데이터 스펙 (전처리팀 preprocess.py 산출물 기준, ebay_laptops_clean_processed.csv):
- 타겟: price_usd (달러 단위)
- feature: brand, model_family, condition_score, release_year, cpu_brand, cpu_family,
  cpu_generation, cpu_suffix, processor_speed_ghz, gpu_vendor, gpu, ram_gb, storage_type,
  ssd_gb, hdd_gb, storage_capacity_gb, has_dual_storage, screen_size_inch,
  resolution_width, resolution_height, os, has_touchscreen, has_backlit_keyboard,
  has_bluetooth, has_webcam, has_wifi
"""

import re
import joblib
import pandas as pd
import numpy as np
# ---------------------------------------------------------------------------
# 상태(condition) 텍스트 -> condition_score 매핑
# preprocess.py의 condition_score() 함수와 동일한 기준을 사용 (학습 데이터와 일치시키기 위함)
#   1: For Parts / Not Working
#   2: 상태 정보 없음(unknown) - 사용자 입력에는 노출하지 않음
#   3: Used
#   4: Refurbished
#   5: Open Box
#   6: New
# ---------------------------------------------------------------------------

CONDITION_SCORE_MAP = {
    "New": 6,
    "Open Box": 5,
    "Refurbished": 4,
    "Used": 3,
    "unknown": 2,
    "For Parts Or Not Working": 1,
}

# ---------------------------------------------------------------------------
# cpu_family -> cpu_brand 자동 추론 (전처리팀 preprocess.py의 cpu_brand() 로직과 동일한 규칙)
# 사용자가 cpu_brand까지 직접 입력하게 하면 번거로우니, cpu_family만 받아서 자동 유추
# ---------------------------------------------------------------------------
def _infer_cpu_brand(cpu_family: str) -> str:
    if cpu_family.startswith("core_") or cpu_family in {"celeron", "pentium", "atom", "xeon", "core_2"}:
        return "intel"
    if cpu_family.startswith("ryzen_") or cpu_family in {"athlon", "amd_a_series", "turion"}:
        return "amd"
    if cpu_family.startswith("apple_"):
        return "apple"
    if cpu_family == "snapdragon":
        return "qualcomm"
    if cpu_family in {"mediatek", "rockchip"}:
        return cpu_family
    return "unknown"

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
        _model = joblib.load("models/price_model_laptop.pkl")
    return _model

# 사용자가 입력한 raw 값을 모델 학습 시 사용된 feature_columns에 맞춰 직접 매핑하는 함수
# refactor: get_dummies를 예측 시점 (아직 행이 1개)에 쓰면 안됨 -> 행이 1개면 범주형 칼럼의 고유값이 항상 1개라서 drop_first=True가 기준값으로 오인.. 칼럼 자체를 삭제함
# -> 모든 입력이 0으로 취급
# 학습 때 만들어진 feature_columns를 기준으로 수동 매핑하기 
def _prepare_input(features: dict, feature_columns: list) -> pd.DataFrame:
    # df_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    df_input = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )
    for key, value in features.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)   # True/False -> 1/0으로 변환
        if key in feature_columns:
            # 숫자형 컬럼(ram_gb, storage_capacity_gb, condition_score 등)은 컬럼명이 그대로 존재
            df_input.at[0, key] = value
        else:
            # 범주형 컬럼(brand, model_family, cpu_family 등) -> "컬럼명_값" 형태로 매핑
            col_name = re.sub(r"[\[\]<>]", "_", f"{key}_{value}")
            if col_name in feature_columns:
                df_input.at[0, col_name] = 1
            # else: 학습 때 없던 값이면 전부 0으로 남음 (모델이 모르는 카테고리)

    return df_input

# ===========================================================================================
# 이 아래 두 함수는 OpenAI API Agent 팀이 직접 가져다 쓸 인터페이스라서 _를 안 붙였습니다
# ===========================================================================================
def predict_price_laptop(
    brand: str,                  # 제조사 (예: dell, hp, lenovo, apple, asus ...)
    model_family: str,           # 제품군 (예: thinkpad, macbook_pro, xps, latitude ...)
    cpu_family: str,             # CPU 등급 (예: core_i5, core_i7, ryzen_5, apple_m1 ...)
    ram_gb: float,                # RAM 용량(GB)
    storage_type: str,           # 저장장치 종류 (ssd, hdd, nvme_ssd, emmc, hybrid)
    storage_capacity_gb: float,   # 저장용량(GB)
    screen_size_inch: float,      # 화면 크기(인치)
    os: str,                     # 운영체제 (windows, macos, chrome_os, linux)
    condition: str,               # 상태 등급 (tools 스키마의 enum 값과 일치해야 함)
    release_year: int = -1,       # 출시연도 (모르면 -1)
) -> dict:
    """
    [ML 모델 호출] 입력받은 노트북 정보를 ML 모델에 전달하여 적정 중고가를 예측한다.
    OpenAI Agent SDK의 function tool(predict_price_laptop)로 그대로 등록되는 함수.
 
    Returns:
        {
            "predicted_price": 355.8,   # 모델이 예측한 적정가격(달러)
            "residual_std": 138.2,      # detect_anomaly()에서 이상 여부 판단 기준으로 사용
        }
    """

    artifact = _get_model()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    residual_std = artifact["residual_std"]

    condition_score = CONDITION_SCORE_MAP.get(condition)
    if condition_score is None:
        raise ValueError(f"알 수 없는 condition 값입니다: {condition!r}")

    raw_features = {
        "brand": brand,
        "model_family": model_family,
        "condition_score": condition_score,
        "release_year": release_year,
        "cpu_brand": _infer_cpu_brand(cpu_family),
        "cpu_family": cpu_family,
        "cpu_generation": "unknown",     # 상세 세대 정보는 사용자 입력에서 노출하지 않음(기본값)
        "cpu_suffix": "unknown",
        "processor_speed_ghz": -1,
        "gpu_vendor": "unknown",
        "gpu": "Unknown",
        "ram_gb": ram_gb,
        "storage_type": storage_type,
        # ssd_gb/hdd_gb는 storage_type 기준으로 단순 분배 (혼합 구성은 사용자 입력에서 미노출)
        "ssd_gb": storage_capacity_gb if storage_type in ("ssd", "nvme_ssd", "emmc") else 0,
        "hdd_gb": storage_capacity_gb if storage_type == "hdd" else 0,
        "storage_capacity_gb": storage_capacity_gb,
        "has_dual_storage": 0,
        "screen_size_inch": screen_size_inch,
        "resolution_width": -1,
        "resolution_height": -1,
        "os": os,
        "has_touchscreen": 0,
        "has_backlit_keyboard": 0,
        "has_bluetooth": 1,   # 최근 노트북 대부분 기본 탑재 -> 합리적 기본값
        "has_webcam": 1,
        "has_wifi": 1,
    }

    X_input = _prepare_input(raw_features, feature_columns)

    # predict()는 배치 예측용이라 결과가 배열로 나옴 -> 입력이 1건이므로 [0]으로 첫 값만 추출
    predicted_price = model.predict(X_input)[0]

    return {
        "predicted_price": round(float(predicted_price), 2),
        "residual_std": float(residual_std),
    }

def detect_anomaly(
    predicted_price: float,   # predict_price_laptop()가 반환한 모델의 적정가 예측값(달러)
    selling_price: float,     # 사용자가 입력한 실제 판매 가격(달러)
    residual_std: float,      # predict_price_laptop()가 함께 반환한 모델 오차의 표준편차(달러)
) -> dict:
    """
    [순수 Python 로직 - 모델 호출 없음]
    예측가격과 판매가격을 비교하여 저가/적정가/고가 여부를 판단한다.
    OpenAI Agent SDK의 function tool(detect_anomaly)로 그대로 등록되는 함수.
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
    # 요건 예시
    price_result = predict_price_laptop(
        brand="dell",
        model_family="latitude",
        cpu_family="core_i5",
        ram_gb=16,
        storage_type="ssd",
        storage_capacity_gb=512,
        screen_size_inch=14.0,
        os="windows",
        condition="Used",
        release_year=2019,
    )
    print("===== predict_price_laptop =====")
    print(price_result)
 
    anomaly_result = detect_anomaly(
        predicted_price=price_result["predicted_price"],
        selling_price=450,
        residual_std=price_result["residual_std"],
    )
    print("===== detect_anomaly =====")
    print(anomaly_result)
