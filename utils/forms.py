import streamlit as st

# 상세검색 아이폰
def iphone_form():

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        model = st.selectbox(
            "기기종류 *",
            [
                "필수 선택 항목.",
                "iPhone 12","iPhone 12 Mini","iPhone 12 Pro","iPhone 12 Pro Max",
                "iPhone 13","iPhone 13 Mini","iPhone 13 Pro","iPhone 13 Pro Max",
                "iPhone 14","iPhone 14 Plus","iPhone 14 Pro","iPhone 14 Pro Max",
                "iPhone 15","iPhone 15 Plus","iPhone 15 Pro","iPhone 15 Pro Max",
                "iPhone 16","iPhone 16 Plus","iPhone 16 Pro","iPhone 16 Pro Max",
                "iPhone 17","iPhone 17 Pro","iPhone 17 Pro Max"
            ]
        )

    with col2:
        storage = st.selectbox(
            "저장용량 *",
            [
                "필수 선택 항목",
                "64GB",
                "128GB",
                "256GB",
                "512GB",
                "1TB",
                "2TB"
            ]
        )

    with col3:
        condition = st.selectbox(
            "제품상태 *",
            [
                "필수 선택 항목",
                "New",
                "Open Box",
                "Used",
                "Excellent - Refurbished",
                "Very Good - Refurbished",
                "Good - Refurbished",
                "For Parts Or Not Working"
            ]
        )

    with col4:
        color = st.selectbox(
            "색상 (선택)",
            [
                "확정x",
                "Space black",
                "Deep purple",
                "Red",
                "White",
                "Blue"
            ]
        )
    with col5 :
        input_price = st.text_input("판매가격")

    return model, storage, condition, color, input_price



# 상세검색 노트북 
def laptop_form():

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.text_input("브랜드")

    with col2:
        model = st.text_input("모델명")

    with col3:
        condition = st.selectbox(
            "상태",
            [
                "선택해주세요",
                "새상품",
                "미개봉 새상품",
                "중고제품"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        os_type = st.selectbox(
            "OS",
            [
                "선택해주세요",
                "Windows",
                "macOS",
                "Linux"
            ]
        )

        #gpu = st.text_input("GPU")

    with col2:
        ram = st.number_input("RAM", 0)

    with col3:
        ssd = st.selectbox(
            "SSD",
            [
                "선택해주세요",
                128,
                256,
                512,
                1024,
                2048,
                4096
            ]
        )

        input_price = st.text_input("판매가격")

    return brand, model, condition, os_type, ram, ssd, input_price
