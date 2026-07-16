"""
currency.py

역할:
- USD <-> KRW 실시간 환율 조회
- Function Calling용 환율 변환

환율은 6시간 동안 캐싱하여 API 호출을 줄인다.
"""

from datetime import datetime, timedelta

import requests

_CACHE_TTL = timedelta(hours=6)

_exchange_rate_cache = {
    "rate": None,
    "date": None,
    "fetched_at": None,
}


def get_usd_krw_rate() -> tuple[float, str]:
    """
    USD -> KRW 환율 조회

    Returns
    -------
    (rate, rate_date)
        rate : 1 USD = ? KRW
    """

    now = datetime.now()

    cached_rate = _exchange_rate_cache["rate"]
    fetched_at = _exchange_rate_cache["fetched_at"]

    # 캐시 사용
    if (
        cached_rate is not None
        and fetched_at is not None
        and (now - fetched_at) < _CACHE_TTL
    ):
        return cached_rate, _exchange_rate_cache["date"]

    # -------------------------------
    # Frankfurter API
    # -------------------------------
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=KRW",
            timeout=5,
        )
        resp.raise_for_status()

        data = resp.json()

        rate = float(data["rates"]["KRW"])
        rate_date = data["date"]

    except Exception as e:
        print(f"[경고] Frankfurter 실패 : {e}")

        # -------------------------------
        # ER API
        # -------------------------------
        try:
            resp = requests.get(
                "https://open.er-api.com/v6/latest/USD",
                timeout=5,
            )
            resp.raise_for_status()

            data = resp.json()

            rate = float(data["rates"]["KRW"])
            rate_date = data.get(
                "time_last_update_utc",
                now.strftime("%Y-%m-%d"),
            )

        except Exception as e2:
            print(f"[경고] ER API 실패 : {e2}")

            # 마지막 fallback
            rate = 1380.0
            rate_date = "Fallback"

    _exchange_rate_cache.update(
        {
            "rate": rate,
            "date": rate_date,
            "fetched_at": now,
        }
    )

    return rate, rate_date


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict:
    """
    Function Calling용

    Parameters
    ----------
    amount : 금액
    from_currency : USD 또는 KRW
    to_currency : USD 또는 KRW
    """

    # 대소문자 통일
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    supported = {"USD", "KRW"}

    if from_currency not in supported:
        raise ValueError(f"지원하지 않는 통화: {from_currency}")

    if to_currency not in supported:
        raise ValueError(f"지원하지 않는 통화: {to_currency}")

    rate, rate_date = get_usd_krw_rate()

    if from_currency == to_currency:
        converted = amount

    elif from_currency == "USD":
        converted = amount * rate

    else:
        converted = amount / rate

    return {
        "original_amount": round(amount, 2),
        "original_currency": from_currency,
        "converted_amount": round(converted, 2),
        "converted_currency": to_currency,
        "rate": round(rate, 4),
        "rate_date": rate_date,
        "is_fallback": rate_date == "Fallback",
    }