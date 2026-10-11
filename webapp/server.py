"""Small local web server wrapping TradingAgentsGraph for a mobile-friendly UI.

Run with: python webapp/server.py
Then open http://<this-pc-ip>:8787 from your phone (same WiFi), or tunnel it
(e.g. with cloudflared) for access from anywhere.
"""
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from webapp.market_brief import compute_market_brief
from webapp.providers import available_providers, list_providers, provider_label, provider_models
from webapp.quick_signal import compute_quick_signal

APP_PASSWORD = os.environ.get("WEBAPP_PASSWORD", "")

app = FastAPI()

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def check_password(x_password: str | None):
    if not APP_PASSWORD:
        return
    if x_password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password")


class AnalyzeRequest(BaseModel):
    ticker: str
    date: str
    providers: list[str] | None = None


class BriefRequest(BaseModel):
    date: str
    providers: list[str] | None = None


def _resolve_providers(requested: list[str] | None) -> list[str]:
    avail = available_providers()
    chosen = [p for p in (requested or []) if p in avail]
    if chosen:
        return chosen
    # No valid selection: fall back to whatever's configured as the default
    # provider if it has a key, else the first available provider.
    default = DEFAULT_CONFIG.get("llm_provider", "").lower()
    if default in avail:
        return [default]
    return avail[:1]


def _run_one_analysis(provider: str, ticker: str, date: str) -> dict:
    try:
        quick_model, deep_model = provider_models(provider)
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = provider
        config["quick_think_llm"] = quick_model
        config["deep_think_llm"] = deep_model
        config["quick_think_provider"] = None
        config["deep_think_provider"] = None
        config["backend_url"] = None
        config["quick_think_backend_url"] = None
        config["deep_think_backend_url"] = None
        ta = TradingAgentsGraph(debug=False, config=config)
        final_state, decision = ta.propagate(ticker, date)
        return {
            "status": "done",
            "decision": decision,
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "trader_investment_plan": final_state.get("trader_investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        }
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        return {"status": "error", "error": str(exc), "trace": traceback.format_exc()}


def _run_job(job_id: str, ticker: str, date: str, providers: list[str]):
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "results": {p: {"label": provider_label(p), "status": "running"} for p in providers},
        }

    def worker(provider: str):
        result = _run_one_analysis(provider, ticker, date)
        with jobs_lock:
            jobs[job_id]["results"][provider] = {"label": provider_label(provider), **result}
            if all(r.get("status") in ("done", "error") for r in jobs[job_id]["results"].values()):
                jobs[job_id]["status"] = "done"

    for p in providers:
        threading.Thread(target=worker, args=(p,), daemon=True).start()


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.get("/api/providers")
def providers(x_password: str | None = Header(default=None)):
    check_password(x_password)
    return {"providers": list_providers()}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, x_password: str | None = Header(default=None)):
    check_password(x_password)
    chosen = _resolve_providers(req.providers)
    if not chosen:
        raise HTTPException(status_code=400, detail="No LLM provider has an API key configured.")
    job_id = uuid.uuid4().hex
    threading.Thread(target=_run_job, args=(job_id, req.ticker.strip().upper(), req.date.strip(), chosen), daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/quick")
def quick(req: AnalyzeRequest, x_password: str | None = Header(default=None)):
    check_password(x_password)
    try:
        return compute_quick_signal(req.ticker.strip().upper(), req.date.strip())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=200)


@app.post("/api/brief")
def brief(req: BriefRequest, x_password: str | None = Header(default=None)):
    check_password(x_password)
    chosen = _resolve_providers(req.providers)
    if not chosen:
        raise HTTPException(status_code=400, detail="No LLM provider has an API key configured.")
    try:
        return compute_market_brief(req.date.strip(), providers=chosen)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=200)


@app.get("/api/status/{job_id}")
def status(job_id: str, x_password: str | None = Header(default=None)):
    check_password(x_password)
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return JSONResponse(job)


if __name__ == "__main__":
    import uvicorn

    if not APP_PASSWORD:
        print("WARNING: WEBAPP_PASSWORD is not set in .env — the server has no password protection.")
    port = int(os.environ.get("PORT", 8787))
    uvicorn.run(app, host="0.0.0.0", port=port)
