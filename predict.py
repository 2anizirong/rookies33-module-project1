"""
예측 함수 (모델 학습 파트에서 관리, OpenAI Agent가 커스텀 툴로 호출)

역할:
- 저장된 models/price_model.pkl 로 적정가격 및 가격범위 예측
- 입력 가격과 비교해 저가 / 적정가 / 고가로 분류
"""

import joblib

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load("models/price_model.pkl")
    return _model


def predict_price_range(features: dict) -> dict:
    """
    OpenAI Agent SDK의 function tool로 등록될 예측 함수

    Args:
        features: {"brand": ..., "model": ..., "ram": ..., "storage": ...,
                   "condition": ..., "input_price": ...}

    Returns:
        {
            "predicted_price": 450000,
            "price_range": [400000, 500000],
            "input_price": 380000,
            "classification": "저가"   # 저가 / 적정가 / 고가
        }
    """
    model = _get_model()
    raise NotImplementedError
