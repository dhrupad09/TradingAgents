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


class BriefRequest(BaseModel):
    date: str


def _run_job(job_id: str, ticker: str, date: str):
    try:
        config = DEFAULT_CONFIG.copy()
        ta = TradingAgentsGraph(debug=False, config=config)
        final_state, decision = ta.propagate(ticker, date)
        with jobs_lock:
            jobs[job_id] = {
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
        with jobs_lock:
            jobs[job_id] = {
                "status": "error",
                "error": str(exc),
                "trace": traceback.format_exc(),
            }


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, x_password: str | None = Header(default=None)):
    check_password(x_password)
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "running"}
    thread = threading.Thread(target=_run_job, args=(job_id, req.ticker.strip().upper(), req.date.strip()), daemon=True)
    thread.start()
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
    try:
        return compute_market_brief(req.date.strip())
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
