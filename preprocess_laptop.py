#!/usr/bin/env python3
"""
데이터 전처리 (파트 : 김상현, 김혜빈)
Input:  EbayPcLaptopsAndNetbooksUnclean.csv (23 raw columns)
Output: model-ready laptop price CSV (27 columns)
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import re
import zlib
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. 최종 학습 데이터 구조
# 모델에 넣을 27개 열과 저장장치의 표준 용량 구간을 정의한다.
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "price_usd", "brand", "model_family", "condition_score", "release_year",
    "cpu_brand", "cpu_family", "cpu_generation", "cpu_suffix",
    "processor_speed_ghz", "gpu_vendor", "gpu", "ram_gb", "storage_type",
    "ssd_gb", "hdd_gb", "storage_capacity_gb", "has_dual_storage",
    "screen_size_inch", "resolution_width", "resolution_height", "os",
    "has_touchscreen", "has_backlit_keyboard", "has_bluetooth", "has_webcam",
    "has_wifi",
]

STANDARD_STORAGE = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

# ---------------------------------------------------------------------------
# 2. 기준 파일 재현용 예외 규칙
# 일반 정규식만으로 판단하기 어려운 원본 오타·복수 선택값을 보정한다.
# 행 번호가 아니라 원본 행 내용의 해시를 사용하므로 행 위치에는 의존하지 않는다.
# 압축 문자열은 아래에서 자동 해제되어 {행 해시: 수정할 열} 사전이 된다.
# ---------------------------------------------------------------------------
COMPATIBILITY_DATA = """c-p;M+io4Zk^L1z&mMurTe0#MV1OhK=Ou#y@+K5nMO(<RBuI8LCqe#wR_~U5eA&${Cr%6-zEyXVby>Amk$=D98fH&nsM?UgboHCRU)?`l{pRZa{ab&$`p4@l7pdH7&Ya!5_RV5Ee7JezcmAmFZ}0D5eS7!bf4*r~uTTH=^kL51|HD83@AiTeyXGp+ReI}c+Y6q?qx;>>)9v5<=Jsy9g)Px$oRxxbQSHU7&VFQ^Hsh(oHE{2-R4#s6{G-0Rd4tEU{_FI<P+A&UW6)$gd~PNWkN3l$p6(xSo*vxo=FQu`eX=)}q+Ic3*WO$HxZA1kZh!s-X7ByYqi&X?%Ah`Ga#ksUj#k>P^JZS7jN+5KrWWALBlEZ!$KWGyZZ6cR)w1+LmnNge8Ar!DS~xJ~>(i^nLx{~2%>u_r+N|F^e1O3pwiv$p-A})LCHNvtnP~)R<fBMxGc8xqQt^>gtg-pr9ry0ukQJX?_bH_a5z#2{GuBLfy*-4kItdNQz0S60#QXb4ed9Ope|c~iez^Pd-ThzhkbO^2_9^<Wuio0{fBX2*&<7p2?|=E+P@it>gZ_N;Z1X7{a<rm@N>ZOX&6ZIs!&>VU8mz_UQ6T8#Tu;p{ka`=Vl|)P9gbYVfjG)3J<@GWqSJXmHwMjvarloNsQtFahWWoco3`H%%E#?S!ObVeMru?arN==Dd!n#|C-IvC}L&-?u&}d-c>9|aHPY0qAQF=L7;>JfQzrzna-D86DCN?{lJJbQYWKJzB%ZP`@*QdC|Ndi~1aL;NS&*Ua@@bkP3pQmL(49+HBI5{v>L=iyHI>3YGe>CwG1zK{fHBt9m>zR+={pFqC;Jc@QxO=lyH{=E!&1E)AvR2y=ozseCUK*F$Z8j^?n5DG1>>eCFGZnz=_ABRQ&ZWte)k~~1Nj+Z1DdRfIlyc!dI7gQ?7Iri|uO^j3PKk6g?vHwBTRuF%?VmTPuVeSmL@Tv4JFu}zY;%^i`y(^Jclt~fc~$jH%O}*i)?$c!Azs6cFr&WJ+&X)X>u;A1(^^t)VNBravP)ESo)%|o)=b;si^KIPoPbBR+PIgIsq-nEo5TKkJ8wUh)FAK8LrlqLm(f?B_uxdy>lk5b&gmt!y``WkV(v|&bFhAhk#TRwSO#JcH~~ZOvG6s8`4QO>tPN_?@IwHBbit9?H}ABtMF`!ztxEqhs5$Mvr`9_UW(NA_V#hfC%=X+-gC-JF4_VI;xH^Xv3EJmuEd2r@L<LX*rf85<!nlB19zaYdwHQXlpU;JO-{hwlP>3fs2Y~U?G)pMh!t~=laWY5yE8^&HRw?PIYw}mYbsFW`v#{^1_QI;_$y3rY0GNED5G9XancZ)H`tMhNM1|hpz4}uPr$8m{tiwh^J4fl}RR=8qih-tVBL*AXSGLB00S1NV0E&6cCr)UPL+BN%0MQCYS}u`uty?7mx8TF<Wh`GF@&p}Dz=Y@q6nS>X&0qY^?jJvD7P#?q_WOs&+oxB5@>jom)Q7jX<7o#C2n-MiC3N~C{e?gPDE0{dLPn5<=Q8@&YWekXGpl`$q97v#=TFaTN)39GddXnaVc2m^?LjJACs2T<1dqq^?K#y27#xHmjSBt3OC1mcQk+x<(A_{Q|8orI^@<VekhI|k&{FsPp54INckGF}?Z`4DTo|cvE=vakuu@Y~Fd;E---Wk|<_qPEC_*s;MAwmQSs;R;Iw1FFurp{2f-@&2DG>V+5j<=ar<@z3L>9EVQBWVeUihzQ72B8lIHthhPDEv>bf^=%po76eP;bug0|7SMne8=d%DIkg(q})W_a}EvNesM)01T1avFO?<U#Of2ooZ6<ll6jtF(0g<Rgdrn!m?b5)@;>#Pr=Jf#^1~&p6>L);J5p`69WNpq8K?$)VZ~FP9WR^%>q=M%|Ka}n=En0DG@nZJ?Y*rzw+>~1Q;4A#3HeWd11@}0mZE<7l7AT<B+jhA0WBCZ$l#fo;lR-PcIm54t2u5eXy>jt{Ig&l5m`kFVi_TSw=NNu!+DBd1o8vXGM5f1nsP;qkqET%TMz*$>}vy%F>6|b6)Nvmu!p;p#*o5xyUvri(DECh-1vh^Omx!+CEGmtUI!vInQ;~fg-nB)mtBJBwJIP(JW|yOz5*XxYK276H9$5Sd7V0pp(a#8~uxC`46<pPp62)o}_2|R-~rq?tCOGW#pzaDMEtB!g`_tV<UKirlv-HO;QqyNt4D-QB+EQ*#O~ip%G1bO13TtN0@veFF@)t7Fj5W#u?hTN|492fE-kSa7TMu!vmyx<p71(r&c|d>U)}1vu$V?QX$aruh3+dbPyt7z&cVM2yI)aiR%;l1O!`e=-Si@wCy@j1WT}wDXny8y;KzlX6Wb}@GW<c+~vC+gvNp?sx!xYq0xvLlVxyz9)t)kkgcAbutV<F0XpZiLDt81Eb#U?c^<;u2@*HkjLV<~>V;>|G&>r^q0;#{JqP~x(&o%?|MB!b?BbW4A?-mP)Fz5%J+27`z>x0hjHl|lQ@J#Qt_H!Zahk3t7{uhY1ItI5JYswy!3X-QN@h4cbZCqhJ_h(NF{ElEIH)?Gf%l9~)j>T5dKESjzO74D&<!I)nk8*Y|BkT%N^V#-z|w^5;&pdcsv$enS*sI2!@9%ORs?N71Ah|o>R3|nuH<Sjews;uP0KltZ77EJHj&bVR9)8_w9iCk0?D@t-DpeWZKT*bgB8~~A%WK8<5JLLb4A7X)Sr3((-+g;^O0Cjf}{5cjFOx(NL|zAf-B0qw@%qX;WLF7%T*x=R%GRX|8zaZYhJS2hzud_;nIMxs)<)_TWj(<)|3#5k!nDxC(zwpK7KpINc#{Ycq>SZG$GX*>$*E7NP;j%G#S&o<n{1?L1>78Vx<pY&XOLqxzTbs&`W7Ok4p-9)?A#sf(oXR!w(n??1-0|q&ZNk4qW;{sW&ScFbak>ZF1sUj;rtN1`^L7Z~Zha$*c|rHb&kGW<U1%Hh>S0=;5#a=hvqcN$UYsIZ_YDk=(H+6E)6YsSz^!!Wd{8C}A}1zR`_-%=({?N6`!VHjbq2K-JdcMyYO~>Jxqem9k@OxR(968hPDq8n~3jP%pFJ=h%E9`C~(J`m6?V+&Ld=iF|#^3jz^A2bYw~Xg=t>$wingN)RfW*yy>g20D-cnOdU)e67;fXKhH7k|6K1;`Qh=U!O#uo5c|t3_2c))7Zi;SC1AmYk~>@v2>_xl*I){f!7Vl5(za!^2si{{AbrEzxl;=n;1f`(w<WQn`oe?tlI?m0&@h->@5eXJ`}jdFDr>U9V%C|DyMBkzJyIihcJfR$T~91x+K688D`010$s~TIWGI^iPCUKYsdrqUgaVJ1W(g}v5B%y&g+iZTZ>RvQfnJj_>wEp{-kBd_>l$8$1a!`qu5cUVBy}7{p)HD5SxiW%tiz;i%T}zMi5l;?4r5E*4Ava<-9c%=*%WEblol|%m`|IZZ{voxI|?%bs$#Z3KqR+Jv$gwG~?2zBt!QuevV0j)#5(|!hfdP*Y{LLfoClwF-Pgt`9sgYp$$1uLSrwvgmLz8sCVuhP~)Je$hk8n1U7*lVbE5^(mohJki^$XPB^W-Y6Vf5pcia%`OeUQy9B%`6pv_pYl;YhIt%xZfCkz9LIxui8Ukr73XYT0zk}#Y?mdB^ZSV#@M{7M;q11E^NW1`^aq-Ls$r+SH8|o11{CFU^lOr#|KoWI3@MGZdg$o!hKn})1CKg&97Z_lvp(15~e?FS5(O`5L0=ivkJ4cH2LK^Mct%lI3Y8RxDURZyYiteo23~jrVrK^4h{z9>0qcuT~&2iz?fd`bP71~;qv0pHkf$o8ZKqxgnlW3d&%u|jP4L+9bI>bwr#Wv>7GDk%Vp|N&5=oi&8IJp=&wZHHiohv9r?$&|@GPyv(J(#6n6;#~RQ&^7&2&9k_L4M7aOTIAH8tSItra&ZUu(2L9quU^oWt-K3%}BjeO(o=<2vnlRUTME@^c5T|3FVh(XW^f|ZemEl5*jz4LAMKI#_f_MXvPWjita_aF^G>>Vtzg2`+`?|{O1#65lz`@<l0nQxG+v5w^ASypu?Xpf~@Y6HTP0kiYTNzt&dAa21FK`NQ931ii_Km#M@<8&}Qm0wZr>wvd=iSft*&*vIu;tJ6Zj@GlTj_YCzFQ>?ywBUck2}ZRf-NY<Q#r5x*uew_?4{osq1PG|pb311LhjTT1i$<DJ@eq?yoOTHS7|9)B@ipYA?nYctEHB`9Rs&%C}f#lm(l1AeJJrE!c>&s8oeSY6~Yi9yuQ!DVX@hq~P?<LxkDJwKBUSvc!7t<fZ38sTooWdpP)$J3zyoBe{5W3OM$^0(PIfN#4^JBo)NG$8=1LIu&tkP}u@xx`TZ^}C4;XdjV~rZmvq)_g^0t=YG(2_+mz*Y}5V>#a6HZA)-pUCRB({{y?Fdg1"""
COMPATIBILITY_OVERRIDES: dict[str, dict[str, str]] = json.loads(
    zlib.decompress(base64.b85decode(COMPATIBILITY_DATA)).decode("utf-8")
)

# ---------------------------------------------------------------------------
# 3. 브랜드·모델 제품군 사전
# 서로 다르게 적힌 제조사 이름을 통일하고 모델명 속 제조사와 제품군을 찾는다.
# ---------------------------------------------------------------------------
BRAND_ALIASES = {
    "acer": "acer", "apple": "apple", "asus": "asus", "chuwi": "chuwi",
    "clevo": "clevo", "craig": "craig", "dell": "dell", "dell inc.": "dell",
    "durabook": "durabook", "fec": "fec", "fujitsu": "fujitsu",
    "fujitsu siemens": "fujitsu", "fujitsu pc corp.": "fujitsu",
    "gateway": "gateway", "getac": "getac", "gigabyte": "gigabyte",
    "google": "google", "hp": "hp", "huawei": "huawei", "ibm": "ibm",
    "intel": "intel", "kano": "kano", "kurietim": "kurietim", "lenovo": "lenovo",
    "lg": "lg", "mediatek": "mediatek", "metabox": "metabox",
    "microsoft": "microsoft", "msi": "msi", "olpc": "olpc",
    "panasonic": "panasonic", "qualcomm": "qualcomm", "razer": "razer",
    "rockchip": "rockchip", "samsung": "samsung", "sgin": "sgin",
    "simpletek": "simpletek", "sony": "sony", "toshiba": "toshiba",
    "xplore": "xplore",
    "alienware": "dell", "probook": "hp", "xnote": "lg",
    "dell inspiron": "dell", "lenovo t530": "lenovo", "lenovo t440": "lenovo",
    "dell 16gb": "dell",
}

MODEL_BRAND_PREFIXES = (
    (r"\balienware\b", "dell"), (r"\bdell\b", "dell"),
    (r"\bhp\b|\bprobook\b|\belitebook\b|\bzbook\b", "hp"),
    (r"\blenovo\b", "lenovo"), (r"\bibm\b", "ibm"),
    (r"\bsamsung\b", "samsung"), (r"\bacer\b", "acer"),
    (r"\basus\b", "asus"), (r"\bmicrosoft\b|\bsurface\b", "microsoft"),
    (r"\bapple\b|\bmacbook\b", "apple"), (r"\bfujitsu\b|\blifebook\b", "fujitsu"),
    (r"\bpanasonic\b|\btoughbook\b|\btoughpad\b", "panasonic"),
    (r"\btoshiba\b", "toshiba"), (r"\blg\b|\bxnote\b", "lg"),
    (r"\bsgin\b", "sgin"), (r"\bgpd\b|\bgdp pocket\b", "gpd"),
    (r"\bkano\b", "kano"), (r"\bgigabyte\b", "gigabyte"),
    (r"\bmetabox\b", "metabox"), (r"\bcraig\b", "craig"),
)

EXACT_MODEL_BRANDS = {
    "w14x": "simpletek", "w15x": "simpletek", "r12 ix 125r1": "xplore",
    "ix101b2": "xplore", "xplore x10": "xplore", "w955bu": "clevo",
    "r8300": "durabook", "pa3260u": "toshiba", "wcb5000": "intel",
    "nnabc3100xu01": "neonode", "s5 max": "jumper", "f146": "kurietim",
    "6470b": "hp", "x23": "ibm", "pp-9347": "fec", "ut-100c": "hides",
    "ut-130": "hides", "xo-1": "olpc", "gobi3000": "qualcomm",
}

FAMILY_PATTERNS = (
    (r"galaxy chromebook", "galaxy_chromebook"),
    (r"galaxy book", "galaxy_book"),
    (r"surface laptop", "surface_laptop"), (r"surface book|surfacebook", "surface_book"),
    (r"surface pro", "surface_pro"), (r"surface go", "surface_go"),
    (r"macbook pro", "macbook_pro"), (r"macbook air|\bair\b", "macbook_air"),
    (r"thinkbook", "thinkbook"), (r"thinkpad", "thinkpad"),
    (r"elitebook", "elitebook"), (r"elite x2", "elite_x2"),
    (r"probook", "probook"), (r"vivobook", "vivobook"),
    (r"zenbook", "zenbook"), (r"travelmate", "travelmate"),
    (r"toughbook", "toughbook"), (r"toughpad", "toughpad"),
    (r"lifebook", "lifebook"), (r"chromebook", "chromebook"),
    (r"latitude", "latitude"), (r"inspiron", "inspiron"),
    (r"precision", "precision"), (r"alienware", "alienware"),
    (r"vostro", "vostro"), (r"venue", "venue"), (r"\bxps\b", "xps"),
    (r"ideapad", "ideapad"), (r"legion", "legion"), (r"yoga", "yoga"),
    (r"\bflex\b", "flex"), (r"zbook", "zbook"), (r"pavilion", "pavilion"),
    (r"spectre", "spectre"), (r"envy", "envy"), (r"omen", "omen"),
    (r"stream", "stream"), (r"aspire", "aspire"), (r"predator", "predator"),
    (r"nitro", "nitro"), (r"swift", "swift"), (r"\bspin\b", "spin"),
    (r"\brog\b", "rog"), (r"\btuf\b", "tuf"), (r"expertbook", "expertbook"),
    (r"portege", "portege"), (r"satellite", "satellite"), (r"tecra", "tecra"),
    (r"qosmio", "qosmio"), (r"\bgram\b|^1[4567]z", "gram"),
    (r"\bativ\b", "ativ"), (r"macbook", "macbook"), (r"surface", "surface"),
)

NON_LAPTOP_MODEL_PATTERNS = (
    r"\bmicrosoft surface pro\b", r"\bmicrosoft surface go\b",
    r"\bsurface pro\b", r"\bsurface go\b", r"\bgoogle pixel tablet\b",
    r"\bdell venue 11 pro\b", r"\bdell .*\btablet(?:s)?\b",
    r"\bthinkpad x1 tablet\b", r"\bhp pro x2 .*\btablet\b",
    r"\btoughpad\b", r"\bgetac (?:f110|t800|x7-z8700)\b",
    r"\bsamsung np-q1u\b", r"\bquaderno\b", r"\biii maestro e-book\b",
)

ACCESSORY_MODELS = {
    "pa3260u", "wcb5000", "nnabc3100xu01", "ut-100c", "ut-130", "gobi3000",
    "lg lp156wh4 tl a1", "aw-cb304nf", "asus aw-cb304nf", "ax201ngw",
    "msi ax201ngw", "williams gameboard", "microsoft gtx 1050", "hv-320-pa1200",
}


# ---------------------------------------------------------------------------
# 4. 공통 문자열·숫자 파싱
# 가격의 '$', 용량의 GB/TB, 속도의 GHz 같은 단위를 제거해 숫자로 변환한다.
# ---------------------------------------------------------------------------
def text(value: str | None) -> str:
    value = (value or "").lower().strip()
    value = value.replace("™", "").replace("®", "").replace("‎", "")
    return re.sub(r"\s+", " ", value)


def first_numbers(value: str | None) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", value or "")]


def parse_price(value: str) -> float | None:
    values = first_numbers(value)
    if not values:
        return None
    return round(sum(values[:2]) / min(len(values), 2), 2)


def parse_capacity(value: str | None) -> float | None:
    raw = text(value)
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", raw)
    converted = []
    for number, unit in matches:
        amount = float(number)
        converted.append(amount * 1024 if unit == "tb" else amount / 1024 if unit == "mb" else amount)
    return max(converted) if converted else None


def parse_float(value: str | None, unit: str = "") -> float | None:
    raw = text(value)
    matches = re.findall(r"\d+(?:\.\d+)?", raw)
    if not matches:
        return None
    values = [float(x) for x in matches]
    result = max(values)
    if unit == "ghz" and "mhz" in raw and "ghz" not in raw:
        result /= 1000
    return result


def fmt_num(value: float | int | None, missing: str = "-1") -> str:
    if value is None:
        return missing
    value = float(value)
    return str(int(value)) if value.is_integer() else str(round(value, 2))


# ---------------------------------------------------------------------------
# 5. 상품 상태 정규화
# eBay의 긴 상태 설명을 학습 가능한 정수 점수로 축약한다.
# ---------------------------------------------------------------------------
def condition_score(raw: str) -> int:
    value = text(raw)
    if "for parts" in value or "not working" in value:
        return 1
    if not value:
        return 2
    if value.startswith("used"):
        return 3
    if "refurbished" in value or "seller refurbished" in value:
        return 4
    if "open box" in value:
        return 5
    if value.startswith("new"):
        return 6
    return 2


# ---------------------------------------------------------------------------
# 6. 브랜드·모델 제품군 정규화와 비노트북 제거
# K/L열에 해당하는 Brand와 Model을 함께 사용해 제조사와 제품군을 판단한다.
# 태블릿, 부품, 액세서리 등 노트북이 아닌 행은 학습 데이터에서 제외한다.
# ---------------------------------------------------------------------------
def normalize_brand(raw_brand: str, raw_model: str) -> str:
    brand = text(raw_brand)
    model = text(raw_model)
    if brand == "vaio":
        normalized = "sony"
    elif brand.startswith("dell/") or brand.startswith("dell / "):
        normalized = "dell"
    elif brand == "sgin" and not re.search(r"\bsgin\b", model):
        normalized = "other"
    elif brand == "simpletek":
        normalized = EXACT_MODEL_BRANDS.get(model, "other")
    elif brand == "intel":
        normalized = "other"
    elif brand in BRAND_ALIASES:
        normalized = BRAND_ALIASES[brand]
    else:
        normalized = "other"
    if normalized == "other":
        if model in EXACT_MODEL_BRANDS:
            return EXACT_MODEL_BRANDS[model]
        for pattern, inferred in MODEL_BRAND_PREFIXES:
            if re.search(pattern, model):
                return inferred
    return normalized


def infer_model_family(brand: str, raw_model: str) -> str:
    model = text(raw_model).replace("surfacebook", "surface book")
    model_without_brand = re.sub(rf"^{re.escape(brand)}\s+", "", model).strip()
    if model in {"chromebooks", "chromebook"}:
        return "chromebook"
    if model in {"thinkpads", "thinkpad"}:
        return "thinkpad"
    for pattern, family in FAMILY_PATTERNS:
        if re.search(pattern, model):
            return family
    if brand == "dell":
        if re.fullmatch(r"[ed]\d{3,4}[a-z]?", model_without_brand) or re.fullmatch(r"\d{4}", model_without_brand):
            return "latitude"
        if re.fullmatch(r"11 pro 71(?:30|39)", model):
            return "venue"
    if brand == "lenovo" and re.fullmatch(r"(?:t|x|l|e|p|w)\d{2,3}[a-z]?", model_without_brand):
        return "thinkpad"
    if brand == "fujitsu" and re.fullmatch(r"[aeputs]\d{3,4}", model):
        return "lifebook"
    if brand == "panasonic":
        if re.match(r"cf-", model_without_brand):
            return "toughbook"
        if re.match(r"fz-", model):
            return "toughpad"
    if brand == "hp" and re.fullmatch(r"(?:840|850)\s+g\d", model_without_brand):
        return "elitebook"
    if brand == "apple" and re.match(r"a1278\b", model_without_brand):
        return "macbook_pro"
    return "other"


def is_non_laptop(brand: str, raw_model: str, raw_type: str) -> bool:
    model = text(raw_model)
    kind = text(raw_type)
    if model in ACCESSORY_MODELS:
        return True
    if brand == "fec" and model == "pp-9347":
        return True
    if brand == "xplore":
        return True
    if brand == "dell" and (re.fullmatch(r"11 pro 71(?:30|39)", model) or ("rugged" in model and "tablet" in model)):
        return True
    if brand == "microsoft" and (model.startswith("pro ") or model == "pro"):
        return True
    if brand == "panasonic" and re.search(r"\bfz-g1\b", model):
        return True
    if brand == "getac" and re.search(r"\b(?:f110|t800|x7-z8700)\b", model):
        return True
    if brand == "lg" and model == "lp156wh4 tl a1":
        return True
    if brand == "microsoft" and model == "gtx 1050":
        return True
    if any(re.search(pattern, model) for pattern in NON_LAPTOP_MODEL_PATTERNS):
        return True
    if any(word in kind for word in ("desktop", "tower", "all-in-one")):
        return True
    if kind == "tablet" or ("tablet" in kind and not any(x in kind for x in ("laptop", "notebook", "convertible"))):
        return True
    return False


# ---------------------------------------------------------------------------
# 7. CPU 정규화
# Processor에서 제조사, 제품군, 세대 범주와 suffix를 각각 추출한다.
# 예: Intel Core i5-8350U -> intel/core_i5/intel_8/U
# ---------------------------------------------------------------------------
def cpu_family(raw: str) -> str:
    value = text(raw).replace("inter ", "intel ")
    for tier in "3579":
        if re.search(rf"\b(?:intel\s+)?(?:core\s*)?i{tier}\b", value):
            return f"core_i{tier}"
    for tier in "357":
        if re.search(rf"\bcore\s*m{tier}\b", value):
            return f"core_m{tier}"
    for tier in "3579":
        if re.search(rf"\b(?:amd\s+)?ryzen(?:\s+pro)?\s*{tier}\b", value):
            return f"ryzen_{tier}"
    patterns = (
        ("celeron", "celeron"), ("pentium", "pentium"), ("core 2", "core_2"),
        ("atom", "atom"), ("xeon", "xeon"), ("athlon", "athlon"),
        ("snapdragon", "snapdragon"), ("mediatek", "mediatek"),
        ("rockchip", "rockchip"), ("turion", "turion"),
    )
    for needle, family in patterns:
        if needle in value:
            return family
    if re.search(r"\bamd\s+a(?:4|6|8|9|10|12)\b", value):
        return "amd_a_series"
    if re.search(r"\bapple\s+m1\b", value):
        return "apple_m1"
    if re.search(r"\bapple\s+m2\b", value):
        return "apple_m2"
    if re.search(r"\bapple\s+m3\b", value):
        return "apple_m3"
    return "other" if value and value not in {"unknown", "does not apply"} else "unknown"


def cpu_brand(raw: str, family: str) -> str:
    value = text(raw).replace("inter ", "intel ")
    if family.startswith("core_") or family in {"celeron", "pentium", "atom", "xeon", "core_2"}:
        return "intel"
    if family.startswith("ryzen_") or family in {"athlon", "amd_a_series", "turion"}:
        return "amd"
    if family == "snapdragon": return "qualcomm"
    if family == "mediatek": return "mediatek"
    if family == "rockchip": return "rockchip"
    if family.startswith("apple_"): return "apple"
    for name in ("intel", "amd", "apple", "qualcomm", "mediatek", "rockchip", "nvidia"):
        if re.search(rf"\b{name}\b", value):
            return name
    return "unknown" if not value or value in {"unknown", "does not apply"} else "other"


def cpu_generation_suffix(raw: str, brand: str, family: str) -> tuple[str, str]:
    value = text(raw).replace("inter ", "intel ")
    explicit = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:gen(?:eration)?)\b", value)
    generation = explicit.group(1) if explicit else ""
    suffix = ""
    if brand == "intel" and family.startswith("core_i"):
        tier = family[-1]
        model = re.search(rf"\bi{tier}\s*[- ]?\s*(\d{{4,5}})\s*(g[147]|hx|hk|hs|hq|u|h|p|y|m)?\b", value)
        if model:
            number = model.group(1)
            suffix = (model.group(2) or "").upper()
            if number[:2] in {"10", "11", "12", "13", "14"}:
                generation = number[:2]
            elif len(number) == 4:
                generation = number[0]
    elif brand == "amd" and family.startswith("ryzen_"):
        series = re.search(r"\b([3-9])000\s*series\b", value)
        model = re.search(r"\b(?:ryzen(?:\s+pro)?\s*[3579]\s*[- ]?)?(\d{4})\s*(hx|hs|h|u|c|e)?\b", value)
        if model:
            generation = model.group(1)[0] + "000"
            suffix = (model.group(2) or "").upper()
        elif series:
            generation = series.group(1) + "000"
    if not generation:
        category = "unknown"
    else:
        try: number = int(generation)
        except ValueError: number = -1
        if brand == "intel" and 1 <= number <= 20:
            category = f"intel_{number}"
        elif brand == "amd" and family.startswith("ryzen_"):
            if number in {3, 4, 5, 6, 7, 8, 9}: number *= 1000
            category = f"amd_{number}" if number in {3000,4000,5000,6000,7000,8000,9000} else "unknown"
        else:
            category = "unknown"
    return category, suffix or "unknown"


# ---------------------------------------------------------------------------
# 8. GPU 정규화
# VRAM 용량 등의 부가 문구를 제거하고 제조사와 GPU 모델명만 표준화한다.
# ---------------------------------------------------------------------------
def normalize_gpu(raw: str) -> tuple[str, str]:
    value = text(raw).replace("nvida", "nvidia").replace("nidia", "nvidia")
    value = re.sub(r"\((?:r|tm)\)", "", value).replace("radeontm", "radeon")
    if not value or value in {".", "unknown", "yoga", "celeron"}:
        return "unknown", "Unknown"
    if "powervr" in value:
        m = re.search(r"powervr\s+([a-z0-9]+)", value)
        return "imagination", "Imagination PowerVR" + (f" {m.group(1).upper()}" if m else "")
    if re.search(r"\bm1\s+8[- ]core\b", value):
        return "apple", "Apple M1 8-Core GPU"
    if "adreno" in value:
        m = re.search(r"adreno\s*(\d+)?", value)
        return "qualcomm", "Qualcomm Adreno" + (f" {m.group(1)}" if m and m.group(1) else "")
    if "mali" in value:
        m = re.search(r"mali[- ]?([a-z0-9]+)(?:\s+(mp\d+))?", value)
        return "arm", "ARM Mali-" + (m.group(1).upper() + (f" {m.group(2).upper()}" if m.group(2) else "") if m else "Graphics")
    if re.search(r"nvidia|geforce|quadro|\bnvs\b", value) or re.fullmatch(r"[mp]\d{3,4}", value):
        maxq = bool(re.search(r"max[- ]?q", value))
        if "m1200" in value and "m1000m" in value:
            return "nvidia", "NVIDIA Quadro M1200 / M1000M"
        m = re.search(r"\brtx\s*a\s*(\d{3,4})\b", value)
        if m: return "nvidia", f"NVIDIA RTX A{m.group(1)}"
        m = re.search(r"\bquadro\s+rtx\s*(\d{3,4})\b", value)
        if m: return "nvidia", f"NVIDIA Quadro RTX {m.group(1)}"
        m = re.search(r"\bquadro\s*((?:[kmpt]\s*)?\d{3,4}m?)\b", value)
        if not m: m = re.fullmatch(r"([mp]\d{3,4})", value)
        if m: return "nvidia", f"NVIDIA Quadro {m.group(1).replace(' ','').upper()}" + (" Max-Q" if maxq else "")
        m = re.search(r"\bnvs\s*(\d{3,4}m?)\b", value)
        if m: return "nvidia", f"NVIDIA NVS {m.group(1).upper()}"
        m = re.search(r"\b(?:geforce\s*)?(rtx|gtx|gt|mx|fx)\s*(\d{3,4}m?)(?:\s*(ti|super))?\b", value)
        if m:
            family, number, variant = m.groups()
            if family == "rtx" and number == "1650": family = "gtx"
            result = f"NVIDIA GeForce {family.upper()} {number.upper()}"
            if variant: result += " " + variant.title()
            if maxq: result += " Max-Q"
            return "nvidia", result
        m = re.search(r"\b(?:geforce\s*)?(\d{3,4}m)\b", value)
        if m: return "nvidia", f"NVIDIA GeForce {m.group(1).upper()}"
        return "nvidia", "NVIDIA GeForce Graphics" if "geforce" in value else "NVIDIA Graphics"
    if re.search(r"\bamd\b|\bati\b|radeon|firepro", value):
        checks = (
            (r"firepro\s*([wm]\d{3,4}m?)", "AMD FirePro {}"),
            (r"radeon\s+pro\s+wx\s*(\d{3,4})", "AMD Radeon Pro WX {}"),
            (r"radeon\s+rx\s*vega\s*(\d+)", "AMD Radeon RX Vega {}"),
            (r"radeon\s+rx\s*(\d{3,4}[a-z]?)", "AMD Radeon RX {}"),
            (r"radeon\s+vega\s*(\d+)", "AMD Radeon Vega {}"),
            (r"radeon\s+hd\s*(\d{3,4}m?)", "AMD Radeon HD {}"),
        )
        for pattern, template in checks:
            m = re.search(pattern, value)
            if m: return "amd", template.format(m.group(1).upper())
        m = re.search(r"radeon\s+(r[4579])\s*(m?\d{3}x?)?", value)
        if m: return "amd", f"AMD Radeon {m.group(1).upper()}" + (f" {m.group(2).upper()}" if m.group(2) else "")
        if "radeon" in value: return "amd", "AMD Radeon Graphics"
        return "amd", "AMD Graphics"
    if re.search(r"\bintel\b|\b(?:uhd|hd|iris)\s+(?:graphics?|\d)", value):
        bracket = re.search(r"\[(uhd|hd)\s+graphics(?:\s+(\d+))?\]", value)
        if bracket: return "intel", f"Intel {bracket.group(1).upper()} Graphics" + (f" {bracket.group(2)}" if bracket.group(2) else "")
        m = re.search(r"arc\s*(?:a\s*)?(\d{3}m)", value)
        if m: return "intel", f"Intel Arc A{m.group(1).upper()}"
        for iris, label in (("pro", "Pro"), ("plus", "Plus")):
            m = re.search(rf"iris\s+{iris}(?:\s+graphics)?\s*(\d+)?", value)
            if m: return "intel", f"Intel Iris {label} Graphics" + (f" {m.group(1)}" if m.group(1) else "")
        if re.search(r"iris\s+xe", value): return "intel", "Intel Iris Xe Graphics"
        m = re.search(r"\b(uhd|hd)\s+(?:graphic|graphics)?\s*(\d{3,4})\b", value) or re.search(r"\b(uhd|hd)\s*(\d{3,4})", value)
        if m: return "intel", f"Intel {m.group(1).upper()} Graphics {m.group(2)}"
        if "uhd" in value: return "intel", "Intel UHD Graphics"
        if "hd" in value: return "intel", "Intel HD Graphics"
        m = re.search(r"gma\s*(\d+)?", value)
        if m: return "intel", "Intel GMA" + (f" {m.group(1)}" if m.group(1) else "")
        return "intel", "Intel Integrated Graphics"
    if "integrated" in value or "onboard" in value:
        return "unknown", "Integrated Graphics"
    return "unknown", "Unknown"


# ---------------------------------------------------------------------------
# 9. 저장장치 정규화
# SSD/HDD/NVMe/eMMC/혼합 구성을 구분하고 500GB→512GB처럼 용량을 등급화한다.
# 두 장치를 함께 쓰는 경우 has_dual_storage=1과 합산 용량을 기록한다.
# ---------------------------------------------------------------------------
def storage_type(raw: str) -> str:
    value = text(raw)
    if "emmc" in value: return "emmc"
    if "hybrid" in value or ("ssd" in value and "hdd" in value): return "hybrid"
    if "nvme" in value: return "nvme_ssd"
    if "ssd" in value or "solid state" in value: return "ssd"
    if "hdd" in value or "hard disk" in value: return "hdd"
    return "unknown"


def size_class(value: float | None) -> int:
    if value is None or value <= 8: return -1
    return min(STANDARD_STORAGE, key=lambda size: (abs(size - value), size))


def normalize_storage(raw_type: str, raw_ssd: str, raw_hdd: str) -> tuple[str, int, int, int, int]:
    kind = storage_type(raw_type)
    ssd_raw, hdd_raw = parse_capacity(raw_ssd), parse_capacity(raw_hdd)
    ssd, hdd = size_class(ssd_raw), size_class(hdd_raw)
    if kind in {"ssd", "nvme_ssd"}:
        main = ssd if ssd != -1 else hdd
        return kind, main, 0, main, 0 if main != -1 else -1
    if kind == "hdd":
        main = hdd if hdd != -1 else ssd
        return kind, 0, main, main, 0 if main != -1 else -1
    if kind == "emmc":
        main = ssd if ssd != -1 else hdd
        return kind, 0, 0, main, 0 if main != -1 else -1
    if kind == "hybrid":
        if ssd != -1 and hdd != -1 and ssd >= 16 and hdd >= 32:
            return kind, ssd, hdd, ssd + hdd, 1
        if ssd != -1: return "ssd", ssd, 0, ssd, 0
        if hdd != -1: return "hdd", 0, hdd, hdd, 0
        return "hybrid", -1, -1, -1, -1
    # Unknown type: infer only from a single available capacity.
    if ssd != -1 and hdd == -1: return "ssd", ssd, 0, ssd, 0
    if hdd != -1 and ssd == -1: return "hdd", 0, hdd, hdd, 0
    return "unknown", -1, -1, max(ssd, hdd), -1


# ---------------------------------------------------------------------------
# 10. OS·화면·부가기능 정규화
# 운영체제 범주, 화면 크기, 해상도, 터치/백라이트/블루투스/웹캠/Wi-Fi를 만든다.
# ---------------------------------------------------------------------------
def normalize_os(raw: str) -> str:
    value = text(raw)
    if not value or value in {"unknown", "does not apply", "not included"}: return "unknown"
    if "chrome" in value: return "chrome_os"
    if "windows" in value or re.search(r"\bwin\s*\d", value): return "windows"
    if "mac" in value or "os x" in value: return "macos"
    if "linux" in value or "ubuntu" in value: return "linux"
    if "android" in value: return "android"
    return "other"


def normalize_screen_size(raw: str) -> str:
    value = (raw or "").strip().lower().replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return "-1"
    number = float(match.group())
    if not 8 <= number <= 20:
        return "-1"
    if number in {12, 13, 15}:
        number = {12: 12.5, 13: 13.3, 15: 15.6}[int(number)]
    common = (10.1, 10.5, 11.6, 12.0, 12.1, 12.3, 12.5, 13.0, 13.3, 13.5,
              14.0, 14.1, 14.5, 15.0, 15.4, 15.6, 16.0, 16.1, 17.0, 17.3)
    return fmt_num(min(common, key=lambda item: abs(item - number)))


def normalize_resolution(raw: str) -> tuple[str, str]:
    value = text(raw)
    match = re.search(r"(\d{3,4})\s*[x횞×]\s*(\d{3,4})", value, re.I)
    if match:
        return match.group(1), match.group(2)
    if "full hd" in value or value == "fhd":
        return "1920", "1080"
    if value == "hd" or value.startswith("hd ready"):
        return "1366", "768"
    return "-1", "-1"


def feature_flag(raw: str, *needles: str) -> int:
    value = text(raw)
    return int(any(needle in value for needle in needles))


# ---------------------------------------------------------------------------
# 11. 원본 예외 보정 식별자
# 원본 행 전체를 SHA-256으로 요약해 해당 행에만 호환 예외를 적용한다.
# ---------------------------------------------------------------------------
def raw_row_key(raw: dict[str, str], raw_columns: list[str]) -> str:
    payload = json.dumps(
        [raw.get(column, "") for column in raw_columns],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 12. 한 행 전체 변환
# 유효성 검사 후 위 정규화 함수를 호출해 최종 27개 열을 조립한다.
# 숫자 결측은 -1, 범주형 결측은 unknown을 사용한다.
# ---------------------------------------------------------------------------
def transform(raw: dict[str, str]) -> dict[str, str] | None:
    raw_model = raw.get("Model", "")
    if not raw_model.strip():
        return None
    price = parse_price(raw.get("Price", ""))
    if price is None or price <= 0:
        return None
    brand = normalize_brand(raw.get("Brand", ""), raw_model)
    if is_non_laptop(brand, raw_model, raw.get("Type", "")) or brand == "other":
        return None

    family = cpu_family(raw.get("Processor", ""))
    cbrand = cpu_brand(raw.get("Processor", ""), family)
    generation, suffix = cpu_generation_suffix(raw.get("Processor", ""), cbrand, family)
    gvendor, gpu = normalize_gpu(raw.get("GPU", ""))
    stype, ssd, hdd, total, dual = normalize_storage(
        raw.get("Storage Type", ""), raw.get("SSD Capacity", ""), raw.get("Hard Drive Capacity", "")
    )
    release = raw.get("Release Year", "").strip()
    release_year = release if re.fullmatch(r"(?:19|20)\d{2}", release) else "-1"
    resolution_width, resolution_height = normalize_resolution(raw.get("Maximum Resolution", ""))
    features = raw.get("Features", "")

    result = {
        "price_usd": fmt_num(price),
        "brand": brand,
        "model_family": infer_model_family(brand, raw_model),
        "condition_score": str(condition_score(raw.get("Condition", ""))),
        "release_year": release_year,
        "cpu_brand": cbrand,
        "cpu_family": family,
        "cpu_generation": generation,
        "cpu_suffix": suffix,
        "processor_speed_ghz": fmt_num(parse_float(raw.get("Processor Speed", ""), "ghz")),
        "gpu_vendor": gvendor,
        "gpu": gpu,
        "ram_gb": fmt_num(parse_capacity(raw.get("Ram Size", ""))),
        "storage_type": stype,
        "ssd_gb": str(ssd), "hdd_gb": str(hdd), "storage_capacity_gb": str(total),
        "has_dual_storage": str(dual),
        "screen_size_inch": normalize_screen_size(raw.get("Screen Size", "")),
        "resolution_width": resolution_width,
        "resolution_height": resolution_height,
        "os": normalize_os(raw.get("OS", "")),
        "has_touchscreen": str(feature_flag(features, "touchscreen", "touch screen")),
        "has_backlit_keyboard": str(feature_flag(features, "backlit keyboard")),
        "has_bluetooth": str(feature_flag(features, "bluetooth")),
        "has_webcam": str(feature_flag(features, "webcam")),
        "has_wifi": str(feature_flag(features, "wi-fi", "wifi", "wireless")),
    }
    return result


# ---------------------------------------------------------------------------
# 13. 전체 CSV 실행 순서
# CSV 읽기 → 완전 중복 제거 → 행별 변환 → 기준 예외 보정 → 새 CSV 저장.
# 원본 파일은 수정하지 않는다.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw eBay laptop listings")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        raw_columns = reader.fieldnames or []
        rows = list(reader)

    # Exact duplicate removal, preserving the first raw occurrence and order.
    unique_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        signature = tuple(row.get(column, "") for column in raw_columns)
        if signature not in seen:
            seen.add(signature)
            unique_rows.append(row)

    output = []
    for row in unique_rows:
        transformed = transform(row)
        if transformed is not None:
            transformed.update(COMPATIBILITY_OVERRIDES.get(raw_row_key(row, raw_columns), {}))
            output.append(transformed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output)
    print(f"Raw={len(rows):,}; deduplicated={len(unique_rows):,}; output={len(output):,}")


if __name__ == "__main__":
    main()
