"""LLM-written daily market brief — one Gemini call, not per-ticker.

Pulls today's % move for major indexes, sectors, and a few macro proxies via
yfinance, then asks the configured "quick think" model to explain the pattern
in plain language. Modeled on the user's existing "Daily Market Brief" Claude
Artifact, adapted to run against this app's own LLM provider instead of a
Robinhood connector.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import yfinance as yf

from tradingagents.llm_clients.factory import create_llm_client
from webapp.providers import available_providers, provider_label, provider_models

INDEXES = [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("DIA", "Dow Jones"), ("IWM", "Russell 2000 (small companies)")]
SECTORS = [
    ("XLK", "Technology"), ("XLC", "Media & telecom"), ("XLY", "Consumer, non-essentials"),
    ("XLF", "Banks & finance"), ("XLV", "Healthcare"), ("XLI", "Industrials"),
    ("XLE", "Energy"), ("XLB", "Materials"), ("XLP", "Consumer staples"),
    ("XLU", "Utilities"), ("XLRE", "Real estate"),
]
MACRO = [("USO", "Oil"), ("GLD", "Gold"), ("TLT", "Long-term bonds")]

ALL_SYMBOLS = [s for s, _ in INDEXES + SECTORS + MACRO]


def _pct_changes(as_of_date: str) -> dict:
    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=12)
    data = yf.download(
        ALL_SYMBOLS,
        start=start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    out = {}
    for sym in ALL_SYMBOLS:
        try:
            closes = data[sym]["Close"].dropna()
        except Exception:
            continue
        if len(closes) < 2:
            continue
        last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
        out[sym] = {"price": last, "chg": (last - prev) / prev * 100}
    return out


def _fmt_pct(v):
    return "unavailable" if v is None else f"{'+' if v > 0 else ''}{v:.2f}%"


def _build_prompt(as_of_date: str, moves: dict) -> str:
    q_lines = "\n".join(f"{name} ({sym}): {_fmt_pct(moves.get(sym, {}).get('chg'))}" for sym, name in INDEXES)
    s_lines = "\n".join(f"{name}: {_fmt_pct(moves.get(sym, {}).get('chg'))}" for sym, name in SECTORS)
    m_lines = "\n".join(f"{name} ({sym}): {_fmt_pct(moves.get(sym, {}).get('chg'))}" for sym, name in MACRO)

    return f"""You write a short daily stock market brief for a small-business owner who is not a finance expert. Use plain, everyday language and explain any finance term in a few words the first time you use it.

You have NO news headlines for this date, so do not state or imply any specific news event, earnings report, Fed decision, or headline as fact. Read the pattern in the numbers instead: which sectors lead and lag, whether safe sectors (utilities, staples, healthcare) or risk-on sectors (tech, consumer discretionary) are winning, whether big or small companies are doing better, and what oil, gold and bonds are doing. Explain what that PATTERN typically suggests, using hedged wording like "this pattern usually suggests" or "this often happens when".

Date: {as_of_date}

MAJOR INDEXES:
{q_lines}

SECTORS:
{s_lines}

OTHER MARKETS:
{m_lines}

Reply with ONLY a JSON object, no markdown fences, in this exact shape:
{{"summary": "2-3 sentences: how the market looks overall on this date",
 "why": ["3-4 short plain-language points about what the pattern in the numbers suggests"],
 "rotation": "2-3 sentences: which parts of the market are leading and lagging, and what that usually means about investor caution vs confidence",
 "watch": "1-2 sentences on what this pattern suggests to keep an eye on next",
 "unknown": "1 sentence making clear this is based on price patterns only, with no news context"}}"""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _run_one_provider(provider: str, prompt: str) -> dict:
    quick_model, _ = provider_models(provider)
    try:
        llm = create_llm_client(provider=provider, model=quick_model).get_llm()
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw, list):
            raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw)
        brief = _parse_json_response(raw)
        return {
            "status": "done",
            "summary": brief.get("summary", ""),
            "why": brief.get("why", []),
            "rotation": brief.get("rotation", ""),
            "watch": brief.get("watch", ""),
            "unknown": brief.get("unknown", ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def compute_market_brief(as_of_date: str, providers: list[str] | None = None) -> dict:
    moves = _pct_changes(as_of_date)
    if not moves:
        return {"status": "error", "error": f"No market data available for {as_of_date}."}

    chosen = [p for p in (providers or available_providers()) if p in available_providers()]
    if not chosen:
        return {"status": "error", "error": "No LLM provider has an API key configured."}

    prompt = _build_prompt(as_of_date, moves)

    def rows(group):
        return [
            {"sym": sym, "name": name, **moves.get(sym, {"price": None, "chg": None})}
            for sym, name in group
        ]

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, len(chosen))) as pool:
        futures = {pool.submit(_run_one_provider, p, prompt): p for p in chosen}
        for future in as_completed(futures):
            p = futures[future]
            results[p] = {"label": provider_label(p), **future.result()}

    return {
        "status": "done",
        "mode": "brief",
        "date": as_of_date,
        "indexes": rows(INDEXES),
        "sectors": rows(SECTORS),
        "macro": rows(MACRO),
        "results": results,
    }
