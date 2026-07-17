# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FRED_API_KEY = os.getenv("FRED_API_KEY")

START = "2015-01-01"

# yfinance tickers
YF_TICKERS = {
    "gold":     "GC=F",      # gold futures — the master series
    "dxy":      "DX-Y.NYB",  # dollar index
    "vix":      "^VIX",      # fear proxy
    "tnx":      "^TNX",      # 10Y nominal yield
    "silver":   "SI=F",
    "usdinr":   "USDINR=X",  # for the INR conversion later
}

# FRED series
FRED_SERIES = {
    "real_yield": "DFII10",  # 10Y TIPS — the important one
    "breakeven":  "T10YIE",  # inflation expectations
}

MASTER = "gold"   # gold's trading days define the index; everything ffills onto it

HORIZON = 5       # trading days ahead — the weekly prediction