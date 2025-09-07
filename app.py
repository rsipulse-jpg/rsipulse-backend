import os, time, threading
from datetime import datetime
from typing import List
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from binance.spot import Spot
import pandas as pd

# -------- Config (env or defaults) --------
QUOTE = os.getenv("QUOTE", "USDC")
INTERVAL = os.getenv("INTERVAL", "15m")
TOPN = int(os.getenv("TOPN", "50"))
MIN_VOL = float(os.getenv("MIN_VOL", "10000000"))  # 10M
RSI_MAX = float(os.getenv("RSI_MAX", "30"))
VOL_MULT = float(os.getenv("VOL_MULT", "1.3"))
EMA_LEN = int(os.getenv("EMA_LEN", "7"))
PERIOD_SEC = int(os.getenv("PERIOD_SEC", "900"))   # 15 min
TIMEOUT = float(os.getenv("TIMEOUT", "10"))

# -------- App --------
app = FastAPI(title="RSIPulse API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def make_client(timeout: float = TIMEOUT) -> Spot:
    return Spot(timeout=timeout)

# -------- Indicators --------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, period: int = 7) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# -------- Core helpers --------
def top_symbols(client: Spot, quote: str, limit: int, min_vol: float) -> List[str]:
    tickers = client.ticker_24hr()
    rows = []
    ban_words = ("UP","DOWN","BULL","BEAR","3L","3S","5L","5S")
    for t in tickers:
        sym = t.get("symbol","")
        if not sym.endswith(quote): continue
        if any(b in sym for b in ban_words): continue
        try:
            vol = float(t.get("quoteVolume",0.0))
        except:
            vol = 0.0
        if vol >= min_vol:
            rows.append((sym, vol))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in rows[:limit]]

def scan_once(quote=QUOTE, interval=INTERVAL, topn=TOPN, min_vol=MIN_VOL,
              rsi_max=RSI_MAX, vol_mult=VOL_MULT, ema_len=EMA_LEN):
    client = make_client()
    syms = top_symbols(client, quote=quote, limit=topn, min_vol=min_vol)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    out = []
    for sym in syms:
        try:
            kl = client.klines(symbol=sym, interval=interval, limit=100)
            if not kl or len(kl) < 30: continue
            opens  = pd.Series([float(k[1]) for k in kl])
            highs  = pd.Series([float(k[2]) for k in kl])
            lows   = pd.Series([float(k[3]) for k in kl])
            closes = pd.Series([float(k[4]) for k in kl])
            vols   = pd.Series([float(k[5]) for k in kl])

            rsi14 = float(rsi(closes, 14).iloc[-1])
            ema_v = float(ema(closes, ema_len).iloc[-1])
            close = float(closes.iloc[-1])

            vol_last = float(vols.iloc[-1])
            vol_avg20 = float(vols[-21:-1].mean()) if len(vols) >= 21 else 0.0
            vol_ok = vol_avg20 > 0 and (vol_last >= vol_mult * vol_avg20)

            note = ""
            if rsi14 < rsi_max and vol_ok and close > ema_v:
                note = "rsi_vol_ema"

            # Panic dump (tight): lower wick ≥45%, close > EMA, RSI <35, vol spike
            h = float(highs.iloc[-1]); l = float(lows.iloc[-1]); o = float(opens.iloc[-1]); c = close
            full = h - l
            low_wick = (o - l) if c >= o else (c - l)
            panic = (full>0) and ((low_wick/full)>=0.45) and (c>ema_v) and (rsi14<35.0) and vol_ok
            if panic:
                note = "panic_dump"

            if note:
                out.append({
                    "time": ts, "symbol": sym, "rsi": round(rsi14,2), "price": close,
                    "vol_last": round(vol_last,2), "vol_avg20": round(vol_avg20,2), "note": note
                })
        except Exception:
            continue
    return out

# -------- Background scheduler (in-memory cache) --------
LAST_RUN = None
LAST_CANDIDATES: List[dict] = []

def scheduler_loop():
    global LAST_RUN, LAST_CANDIDATES
    while True:
        try:
            LAST_CANDIDATES = scan_once()
            LAST_RUN = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            pass
        time.sleep(PERIOD_SEC)

@app.on_event("startup")
def on_start():
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()

# -------- API --------
class ScanResponse(BaseModel):
    time: str
    candidates: List[dict]

@app.get("/health")
def health():
    return {"ok": True, "last_run": LAST_RUN, "count": len(LAST_CANDIDATES)}

@app.get("/symbols")
def symbols(quote: str = Query(QUOTE), topn: int = TOPN, min_vol: float = MIN_VOL):
    client = make_client()
    return {"quote": quote, "symbols": top_symbols(client, quote, topn, min_vol)}

@app.get("/scan", response_model=ScanResponse)
def scan_endpoint(quote: str = Query(QUOTE), interval: str = INTERVAL, topn: int = TOPN,
                  min_vol: float = MIN_VOL, rsi_max: float = RSI_MAX,
                  vol_mult: float = VOL_MULT, ema_len: int = EMA_LEN):
    cands = scan_once(quote, interval, topn, min_vol, rsi_max, vol_mult, ema_len)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return {"time": ts, "candidates": cands}

@app.get("/alerts", response_model=ScanResponse)
def alerts():
    return {"time": LAST_RUN or "", "candidates": LAST_CANDIDATES}

