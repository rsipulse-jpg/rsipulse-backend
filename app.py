# app.py — RSI Pulse Backend (Free + Premium modes, Demo preserved)
# FastAPI + binance-connector (public endpoints only)

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from binance.spot import Spot

# ---------------- Env-config (safe defaults; override on Render) ----------------
QUOTE = os.getenv("QUOTE", "USDC")                 # scan pairs ending with this quote (e.g., USDC or USDT)
INTERVAL = os.getenv("INTERVAL", "5m")             # klines timeframe
TOPN = int(os.getenv("TOPN", "30"))                # take top-N symbols by 24h quote volume
MIN_VOL = float(os.getenv("MIN_VOL", "1000000"))   # liquidity floor (24h quote volume), default $1M

# FREE mode — super basic, RSI only (period + threshold), uses liquidity filter
FREE_RSI_PERIOD = int(os.getenv("FREE_RSI_PERIOD", "14"))
FREE_RSI_THRESHOLD = float(os.getenv("FREE_RSI_THRESHOLD", "70"))  # e.g., 40 is conservative “weak/oversold”

# PREMIUM mode — richer triggers (RSI + EMA + Volume) + tight panic-dump
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_THRESHOLD = float(os.getenv("RSI_THRESHOLD", os.getenv("RSI_MAX", "50")))  # keep backward compat with RSI_MAX
VOL_MULT = float(os.getenv("VOL_MULT", "1.0"))      # last volume must be >= this × avg20
EMA_LEN = int(os.getenv("EMA_LEN", "7"))            # EMA length for close>EMA check
PANIC_WICK_MIN = float(os.getenv("PANIC_WICK_MIN", "0.45"))  # long lower wick ratio
PANIC_RSI_MAX = float(os.getenv("PANIC_RSI_MAX", "35.0"))    # RSI upper bound for panic
TIMEOUT = float(os.getenv("TIMEOUT", "10"))         # not used by public endpoints, kept for compat

# Optional Binance API keys (NOT required for public endpoints)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ---------------- FastAPI app ----------------
app = FastAPI(title="RSI Pulse Backend", version="1.1.0")

# ---------------- Utilities ----------------
def make_client() -> Spot:
    # Public data works fine without keys; keys only used if provided
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        return Spot(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    return Spot()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def top_symbols(client: Spot, quote: str, limit: int, min_vol: float) -> List[str]:
    tickers = client.ticker_24hr()
    rows = []
    ban_words = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")  # exclude leveraged tokens
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith(quote):
            continue
        if any(b in sym for b in ban_words):
            continue
        try:
            vol = float(t.get("quoteVolume", 0.0))
        except Exception:
            vol = 0.0
        if vol >= min_vol:
            rows.append((sym, vol))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:limit]]

# ---------------- Demo data ----------------
def demo_candidates(ts: str):
    return [
        {"time": ts, "symbol": "BTCUSDC", "rsi": 28.4, "price": 112345.12,
         "vol_last": 12000.0, "vol_avg20": 8900.0, "note": "rsi_vol_ema"},
        {"time": ts, "symbol": "SOLUSDC", "rsi": 31.7, "price": 206.45,
         "vol_last": 75000.0, "vol_avg20": 52000.0, "note": "panic_dump"},
        {"time": ts, "symbol": "LINKUSDC", "rsi": 29.9, "price": 25.66,
         "vol_last": 18500.0, "vol_avg20": 12300.0, "note": "rsi_vol_ema"},
    ]

# ---------------- Scan logic ----------------
def scan_once(
    mode: str,
    quote: str = QUOTE,
    interval: str = INTERVAL,
    topn: int = TOPN,
    min_vol: float = MIN_VOL,
    free_rsi_period: int = FREE_RSI_PERIOD,
    free_rsi_threshold: float = FREE_RSI_THRESHOLD,
    rsi_period: int = RSI_PERIOD,
    rsi_threshold: float = RSI_THRESHOLD,
    vol_mult: float = VOL_MULT,
    ema_len: int = EMA_LEN,
) -> List[dict]:
    """
    Modes:
      - free: minimal RSI-only gate (period+threshold) + liquidity
      - premium: RSI + EMA + volume confirmation; tight panic-dump override
    """
    client = make_client()
    syms = top_symbols(client, quote=quote, limit=topn, min_vol=min_vol)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    out = []

    for sym in syms:
        try:
            kl = client.klines(symbol=sym, interval=interval, limit=100)
            if not kl or len(kl) < 30:
                continue

            # Build series
            opens  = pd.Series([float(k[1]) for k in kl])
            highs  = pd.Series([float(k[2]) for k in kl])
            lows   = pd.Series([float(k[3]) for k in kl])
            closes = pd.Series([float(k[4]) for k in kl])
            vols   = pd.Series([float(k[5]) for k in kl])

            close = float(closes.iloc[-1])

            # Common: 20-bar avg volume (excluding current)
            vol_last = float(vols.iloc[-1])
            vol_avg20 = float(vols[-21:-1].mean()) if len(vols) >= 21 else 0.0
            vol_ok = (vol_avg20 > 0) and (vol_last >= vol_mult * vol_avg20)  # used in premium and panic

            note = ""
            rsi_used = None

            if mode == "free":
                # FREE: RSI-only gate, keep it simple by design
                rsi_val = float(rsi(closes, free_rsi_period).iloc[-1])
                rsi_used = rsi_val
                if rsi_val < free_rsi_threshold:
                    note = "rsi_basic"

            elif mode == "premium":
                # PREMIUM: fuller logic
                rsi_val = float(rsi(closes, rsi_period).iloc[-1])
                rsi_used = rsi_val
                ema_v = float(ema(closes, ema_len).iloc[-1])

                # Base gate: RSI + VOL + close>EMA
                if (rsi_val < rsi_threshold) and vol_ok and (close > ema_v):
                    note = "rsi_vol_ema"

                # Tight panic-dump: long lower wick, low RSI, vol_ok, close>EMA
                h = float(highs.iloc[-1]); l = float(lows.iloc[-1]); o = float(opens.iloc[-1]); c = close
                full = h - l
                low_wick = (o - l) if c >= o else (c - l)
                if (full > 0):
                    wick_ratio = low_wick / full
                    panic = (wick_ratio >= PANIC_WICK_MIN) and (c > ema_v) and (rsi_val < PANIC_RSI_MAX) and vol_ok
                    if panic:
                        note = "panic_dump"

            # Add candidate if any rule fired
            if note:
                out.append({
                    "time": ts,
                    "symbol": sym,
                    "rsi": round(rsi_used if rsi_used is not None else 0.0, 2),
                    "price": close,
                    "vol_last": round(vol_last, 2),
                    "vol_avg20": round(vol_avg20, 2),
                    "note": note
                })

        except Exception:
            # Skip symbol on any error (network/data issues)
            continue

    return out

# ---------------- In-memory cache for /alerts ----------------
LAST_RUN: Optional[str] = None
LAST_CANDIDATES: List[dict] = []

# ---------------- Models ----------------
class Candidate(BaseModel):
    time: str
    symbol: str
    rsi: float
    price: float
    vol_last: float
    vol_avg20: float
    note: str

class ScanResponse(BaseModel):
    time: str
    candidates: List[Candidate]

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

@app.get("/symbols")
def symbols(
    quote: str = Query(QUOTE),
    topn: int = Query(TOPN, ge=1, le=500),
    min_vol: float = Query(MIN_VOL, ge=0),
) -> List[str]:
    client = make_client()
    return top_symbols(client, quote=quote, limit=topn, min_vol=min_vol)

@app.get("/scan", response_model=ScanResponse)
def scan_endpoint(
    mode: str = Query("free", pattern="^(free|premium)$"),
    quote: str = Query(QUOTE),
    interval: str = INTERVAL,
    topn: int = TOPN,
    min_vol: float = MIN_VOL,
    # Free controls
    free_rsi_period: int = FREE_RSI_PERIOD,
    free_rsi_threshold: float = FREE_RSI_THRESHOLD,
    # Premium controls
    rsi_period: int = RSI_PERIOD,
    rsi_threshold: float = RSI_THRESHOLD,
    vol_mult: float = VOL_MULT,
    ema_len: int = EMA_LEN,
    test: int = 0,  # keep demo hook
):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    if test == 1:
        cands = demo_candidates(ts)
    else:
        cands = scan_once(
            mode=mode,
            quote=quote,
            interval=interval,
            topn=topn,
            min_vol=min_vol,
            free_rsi_period=free_rsi_period,
            free_rsi_threshold=free_rsi_threshold,
            rsi_period=rsi_period,
            rsi_threshold=rsi_threshold,
            vol_mult=vol_mult,
            ema_len=ema_len,
        )

    # update cache for /alerts
    global LAST_RUN, LAST_CANDIDATES
    LAST_RUN = ts
    LAST_CANDIDATES = cands

    return {"time": ts, "candidates": cands}

@app.get("/alerts", response_model=ScanResponse)
def alerts(test: int = 0):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    if test == 1:
        return {"time": ts, "candidates": demo_candidates(ts)}
    return {"time": LAST_RUN or "", "candidates": LAST_CANDIDATES}

