"""
예측 함수 (모델 학습 파트에서 관리, OpenAI Agent가 커스텀 툴로 호출)

역할:
- 저장된 models/price_model.pkl 로 적정가격 및 가격범위 예측
- 입력 가격과 비교해 저가 / 적정가 / 고가로 분류
"""

import joblib
import pandas as pd
import re

# 모델을 매번 파일에서 다시 불러오지 않도록 캐싱해두는 전역 변수
# (처음 한 번만 로드하고, 이후 호출부터는 메모리에 있는 걸 재사용)
# _이걸로 이름 앞에 붙이면 내부용이라는 관례적 신호라고 합니다! 외부에서 건들이지 않아야 한다는 거래요.
_model = None


def _get_model():
    """
    models/price_model.pkl 을 불러와서 반환하는 함수.
    파일 안에는 모델뿐만 아니라 학습 시 사용한 컬럼 목록(feature_columns),
    잔차 표준편차(residual_std: 모델이 평소에 얼마나 틀리는지)도 같이 딕셔너리 형태로 저장되어 있음.
    """
    global _model
    if _model is None:
        # 처음 호출될 때만 실제로 디스크에서 파일을 읽어옴
        _model = joblib.load("models/price_model.pkl")
    return _model


def _prepare_input(features: dict, feature_columns: list) -> pd.DataFrame:
    """
    사용자가 입력한 raw 값(dict)을 모델이 학습했던 것과 동일한 형태(원핫인코딩 된 컬럼 구조)로 변환하는 함수.
    예: {"brand": "Apple"} -> 학습 때 brand_Apple, brand_Samsung ... 컬럼이 있었다면
        그 구조에 맞춰 brand_Apple=1, brand_Samsung=0 형태로 바꿔줌
    """
    # 입력 dict 하나를 표 형태(1행짜리 DataFrame)로 변환
    df_input = pd.DataFrame([features])

    # 문자열(범주형) 컬럼 찾기 (예: brand, model_family 등)
    cat_cols = df_input.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        # 학습 때와 동일한 방식(get_dummies)으로 원핫인코딩 수행
        df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

    # XGBoost는 컬럼명에 [ ] < > 같은 특수문자가 있으면 에러가 나므로
    # 학습 때와 동일하게 특수문자를 언더스코어(_)로 치환
    # 지금은 rf 로 임의로 해두긴 했는데 xgb 할 수도 있으니까 추가 해둘겠습니다. 
    df_input.columns = [re.sub(r"[\[\]<>]", "_", col) for col in df_input.columns]

    # 학습 때 있던 컬럼 구조(feature_columns)에 맞춰 정렬
    # - 입력값엔 없지만 학습 때 있던 컬럼(예: 한 번도 안 나온 브랜드) -> 0으로 채움
    # - 컬럼 순서도 학습 때와 동일하게 맞춰야 모델이 값을 올바르게 해석함
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    return df_input


def _classify(input_price: float, lower: float, upper: float) -> str:
    """
    사용자가 입력한 실제 판매가격(input_price)을
    예측된 가격 범위(lower~upper)와 비교해서 저가/적정가/고가로 분류. 
    """
    if input_price < lower:
        return "저가"      # 예측 범위보다 낮음 -> 시세보다 싸게 나온 매물
    elif input_price > upper:
        return "고가"      # 예측 범위보다 높음 -> 시세보다 비싸게 나온 매물(바가지 의심)
    else:
        return "적정가"    # 예측 범위 안에 있음 -> 합리적인 가격대

# 이건 OpenAI API Agent 팀이 직접 가져다 쓸 인터페이스라서 _를 처리하지 않았습니당 
def predict_price_range(features: dict) -> dict:
    """
    OpenAI Agent SDK의 function tool로 등록될 예측 함수.
    에이전트가 사용자 입력을 받아 이 함수를 호출하면,
    적정가격과 가격범위, 저가/적정가/고가 분류 결과를 반환함.

    Args:
        features: {"brand": ..., "model": ..., "ram": ..., "storage": ...,
                   "condition": ..., "input_price": ...}
                   * 실제 key 이름은 학습에 쓴 feature와 동일해야 함
                   * "input_price"는 모델의 예측 입력값이 아니라,
                     예측 결과와 비교해서 분류하는 용도로만 쓰임 (모델에는 안 들어감)

    Returns:
        {
            "predicted_price": 450000,     # 모델이 예측한 적정가격
            "price_range": [400000, 500000],  # 적정가로 인정되는 가격 범위
            "input_price": 380000,          # 사용자가 실제로 입력한 판매가격
            "classification": "저가"        # 저가 / 적정가 / 고가
        }
    """
    # 모델 + 부가정보(학습 컬럼, 잔차 표준편차) 불러오기
    artifact = _get_model()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    residual_std = artifact["residual_std"]

    # input_price는 모델 예측에 쓰는 값이 아니므로 따로 분리
    input_price = features.get("input_price")
    model_input = {k: v for k, v in features.items() if k != "input_price"}

    # 나머지 feature들만 모델이 이해할 수 있는 형태로 변환
    X_input = _prepare_input(model_input, feature_columns)

    # 모델로 적정가격 예측 (배열로 나오므로 [0]으로 첫 값만 꺼냄)
    predicted_price = float(model.predict(X_input)[0])

    # 예측값 주변에 '허용 범위'를 잡음
    # z값이 클수록 범위가 넓어져서 '적정가' 판정이 관대해짐
    # z=1.0(엄격) ~ z=2.0(관대) 사이에서 실제 데이터 보면서 조정하면 됨
    z = 1.5
    lower = predicted_price - z * residual_std
    upper = predicted_price + z * residual_std

    # 사용자가 입력한 실제 가격과 비교해서 저가/적정가/고가 판정
    # input_price가 없는 경우(가격 미입력)는 분류를 건너뜀
    classification = _classify(input_price, lower, upper) if input_price is not None else None

    # 에이전트가 바로 활용할 수 있는 형태(dict)로 결과 반환
    return {
        "predicted_price": round(predicted_price, 2),
        "price_range": [round(lower, 2), round(upper, 2)],
        "input_price": input_price,
        "classification": classification,
    }