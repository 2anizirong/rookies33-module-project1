"""
데이터 전처리 담당 (전처리 파트: 김상현, 김혜빈)

역할:
- 원본 eBay Unclean CSV를 읽어서 결측치/이상값 처리
- 브랜드, 모델, RAM, 저장공간, 상태 등 feature 생성
- 학습에 쓸 정제된 데이터를 data/processed/ 에 저장
"""

import pandas as pd


def load_raw_data(path: str = "data/raw/EbayPcLaptopsAndNetbooksUnclean.csv") -> pd.DataFrame:
    """원본 CSV 로드"""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 처리, 가격 문자열 → 숫자 변환, 이상값 제거 등"""
    raise NotImplementedError


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """브랜드/모델/RAM/저장공간/상태 등 모델 학습용 feature 생성"""
    raise NotImplementedError


if __name__ == "__main__":
    df = load_raw_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df.to_csv("data/processed/laptops_clean.csv", index=False)
    print(f"전처리 완료: {len(df)}행 저장됨")
