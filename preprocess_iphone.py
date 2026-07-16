"""
중고 아이폰 판매 데이터를 모델 학습에 적합한 형태로 전처리.

주요 처리 과정:
- 저장공간 컬럼 통합 및 결측값 복구
- 저장공간을 복구할 수 없는 행 제거
- 제품 상태를 condition_score로 점수화
- 상품 제목에서 색상 정보 추출
- 색상값을 소문자로 통일하고 결측값을 unknown으로 처리
- 모델 학습에 사용하지 않는 컬럼 제거
- 최종 결과를 iphone_clean_processed.csv로 저장
"""


import argparse
import csv
import re
from pathlib import Path

# 상품 상태를 좋은 상태(7점)부터 부품용(1점)까지 순서형 점수로 변환한다.
CONDITION_SCORES = {
    "New": 7,
    "Open Box": 6,
    "Excellent - Refurbished": 5,
    "Very Good - Refurbished": 4,
    "Good - Refurbished": 3,
    "Used": 2,
    "For Parts Or Not Working": 1,
}

# 제목에서 인식할 iPhone 색상 목록이다.
# 여러 색상이 있는 제목은 제목에서 가장 먼저 등장하는 색상을 사용한다.
COLOR_RULES = (
    r"\bnatural\s+titanium\b",
    r"\bnavy\b",
    r"\bgraphite\b",
    r"\borange\b",
    r"\bsilver\b",
    r"\bpink\b",
    r"\bgreen\b",
    r"\byellow\b",
    r"\bpurple\b",
    r"\bred\b",
    r"\bblue\b",
    r"\bwhite\b",
    r"\bblack\b",
    r"\bstarlight\b",
    r"\bmidnight\b",
    r"\bgold\b",
)
COLOR_PATTERN = re.compile(
    "|".join(f"(?:{pattern})" for pattern in COLOR_RULES),
    flags=re.IGNORECASE,
)

# 모델 학습에 사용할 최종 컬럼만 지정한다.
# title(A열)은 전처리에만 사용한다.
# 모델에 쓰지 않는 기존 I~O열과 lastUpdated도 최종 파일에서 제외한다.
OUTPUT_FIELDS = [
    "model_family",
    "generation_number",
    "is_pro",
    "storage_gb",
    "condition",
    "condition_score",
    "price",
    "color",
]


def compact_number(value: str) -> str:
    """128.0 같은 저장공간 값을 128 형태의 정수 문자열로 정리한다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return str(int(number)) if number.is_integer() else str(number)


def recover_storage(title: str) -> str:
    """제목에 저장공간 후보가 정확히 하나일 때만 안전하게 복구한다."""
    gb_values = re.findall(r"(?<!\d)(64|128|256|512)(?!\d)", title)
    tb_values = re.findall(r"(?i)(?<!\d)(1|2)\s*TB(?![A-Za-z])", title)
    values = [int(value) for value in gb_values]
    values.extend(int(value) * 1024 for value in tb_values)
    return str(values[0]) if len(values) == 1 else ""


def extract_color(title: str) -> str:
    """제목의 첫 번째 색상을 소문자로 추출하고, 없으면 unknown을 반환한다."""
    match = COLOR_PATTERN.search(title)
    return match.group(0).lower() if match else "unknown"


def main() -> None:
    """명령줄에서 받은 원본 CSV를 전처리하여 지정한 경로에 저장한다."""
    parser = argparse.ArgumentParser(
        description="Preprocess iPhone resale data for model training"
    )
    parser.add_argument("input_csv", type=Path, help="전처리할 원본 CSV 경로")
    parser.add_argument("output_csv", type=Path, help="전처리 결과 CSV 경로")
    args = parser.parse_args()

    # 입력 파일과 필수 헤더가 있는지 실행 초기에 확인한다.
    if not args.input_csv.is_file():
        parser.error(f"입력 파일을 찾을 수 없습니다: {args.input_csv}")

    # 출력 폴더가 없으면 자동으로 생성한다.
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0

    # 원본 CSV를 한 행씩 읽어 메모리 사용량을 줄인다.
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_fields = {
            "title",
            "model_family",
            "generation_number",
            "is_pro",
            "storage_gb_numeric",
            "condition",
            "price",
        }
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                "원본 CSV에 필수 컬럼이 없습니다: " + ", ".join(sorted(missing_fields))
            )

        # Excel에서도 한글이 깨지지 않도록 UTF-8 BOM 형식으로 저장한다.
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for row in reader:
                # 숫자형 저장공간을 우선 사용하고, 비어 있으면 제목에서 복구한다.
                storage = compact_number(row.get("storage_gb_numeric", ""))
                if not storage:
                    storage = recover_storage(row.get("title", ""))

                # 저장공간을 확실하게 알 수 없는 행은 모델 학습 데이터에서 제거한다.
                if not storage:
                    continue

                # 필요한 원본값과 새로 만든 파생변수만 최종 행에 담는다.
                cleaned = {
                    "model_family": row.get("model_family", ""),
                    "generation_number": row.get("generation_number", ""),
                    "is_pro": row.get("is_pro", ""),
                    "storage_gb": storage,
                    "condition": row.get("condition", ""),
                    "condition_score": CONDITION_SCORES.get(row.get("condition", ""), ""),
                    "price": row.get("price", ""),
                    "color": extract_color(row.get("title", "")),
                }
                writer.writerow(cleaned)
                written_rows += 1

    print(f"Wrote {written_rows:,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
