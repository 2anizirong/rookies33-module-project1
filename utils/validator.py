# 아이폰 유효성 검사
def validate_iphone(model, storage, condition):

    errors = []

    if model == "필수 선택항목입니다.":
        errors.append("📱 기기종류를 선택해주세요.")

    if storage == "필수 선택항목입니다":
        errors.append("💾 저장용량을 선택해주세요.")

    if condition == "선택해주세요":
        errors.append("📦 제품 상태를 선택해주세요.")

    return errors



# 노트북 유효성 검사
def validate_laptop(brand, model, condition):

    errors = []

    if brand.strip() == "":
        errors.append("💻 브랜드를 입력해주세요.")

    if model.strip() == "":
        errors.append("💻 모델명을 입력해주세요.")

    if condition == "선택해주세요":
        errors.append("📦 제품 상태를 선택해주세요.")

    return errors