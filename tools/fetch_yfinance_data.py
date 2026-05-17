import yfinance as yf
import pandas as pd
import numpy as np
import os

print("1. 開始從 yfinance 下載資料...")
ticker = "NVDA"
df = yf.download(ticker, start="2016-01-01", end="2026-01-01")

df.columns = df.columns.droplevel('Ticker')
df.columns.name = None
df = df.reset_index()

df.rename(columns={"Date": "date"}, inplace=True)

df = df[["date", "Open", "High", "Low", "Close"]]

df.to_csv("dataset/NVDA-2016-2025.csv", index=False)
