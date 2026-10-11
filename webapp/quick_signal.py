"""Instant, zero-cost technical signal — no LLM calls.

Pure arithmetic on cached OHLCV data (RSI/MACD/SMA/Bollinger), scored into a
5-tier rating. This is a mechanical heuristic, not the multi-agent LLM
analysis — it trades depth for speed (sub-second) and cost (free, no API
calls beyond the Yahoo Finance data fetch).
"""
from stockstats import wrap

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

RATING_ORDER = ["Sell", "Underweight", "Hold", "Overweight", "Buy"]


def _latest_row(ticker: str, as_of_date: str):
    data = load_ohlcv(ticker, as_of_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    for col in ("close_50_sma", "close_200_sma", "rsi", "macd", "macds", "boll_ub", "boll_lb"):
        df[col]  # trigger stockstats calculation
    as_of_str = str(as_of_date)
    matching = df[df["Date"].str.startswith(as_of_str)]
    row = matching.iloc[-1] if not matching.empty else df.iloc[-1]
    return row


def compute_quick_signal(ticker: str, as_of_date: str) -> dict:
    row = _latest_row(ticker, as_of_date)

    close = float(row["close"])
    sma50 = float(row["close_50_sma"])
    sma200 = float(row["close_200_sma"])
    rsi = float(row["rsi"])
    macd = float(row["macd"])
    macds = float(row["macds"])
    boll_ub = float(row["boll_ub"])
    boll_lb = float(row["boll_lb"])

    score = 0
    reasons = []

    if close > sma50:
        score += 1
        reasons.append(f"Price (${close:.2f}) is above the 50-day average (${sma50:.2f}) — short/medium-term uptrend.")
    else:
        score -= 1
        reasons.append(f"Price (${close:.2f}) is below the 50-day average (${sma50:.2f}) — short/medium-term downtrend.")

    if sma50 > sma200:
        score += 1
        reasons.append("50-day average is above the 200-day average — long-term trend is bullish (golden-cross territory).")
    else:
        score -= 1
        reasons.append("50-day average is below the 200-day average — long-term trend is bearish.")

    if macd > macds:
        score += 1
        reasons.append("MACD is above its signal line — bullish momentum.")
    else:
        score -= 1
        reasons.append("MACD is below its signal line — bearish momentum.")

    if rsi < 30:
        score += 1
        reasons.append(f"RSI ({rsi:.1f}) is oversold — room to bounce.")
    elif rsi > 70:
        score -= 1
        reasons.append(f"RSI ({rsi:.1f}) is overbought — stretched to the upside.")
    else:
        reasons.append(f"RSI ({rsi:.1f}) is neutral — no momentum extreme either way.")

    if close < boll_lb:
        score += 1
        reasons.append("Price is below the lower Bollinger Band — statistically stretched low.")
    elif close > boll_ub:
        score -= 1
        reasons.append("Price is above the upper Bollinger Band — statistically stretched high.")

    # score range is roughly -4..+4; map to the 5-tier rating
    if score >= 3:
        rating = "Buy"
    elif score >= 1:
        rating = "Overweight"
    elif score <= -3:
        rating = "Sell"
    elif score <= -1:
        rating = "Underweight"
    else:
        rating = "Hold"

    return {
        "status": "done",
        "mode": "quick",
        "decision": rating,
        "score": score,
        "final_trade_decision": "\n".join(f"- {r}" for r in reasons)
        + "\n\nMechanical technical-indicator heuristic only — no AI reasoning, no fundamentals/news/sentiment. "
        + "Not financial advice.",
        "price_snapshot": {
            "close": close, "sma50": sma50, "sma200": sma200,
            "rsi": rsi, "macd": macd, "macds": macds,
            "boll_ub": boll_ub, "boll_lb": boll_lb,
        },
    }
