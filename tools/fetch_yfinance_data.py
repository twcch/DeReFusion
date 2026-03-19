import yfinance as yf
import pandas as pd
import numpy as np
import os

print("1. 開始從 yfinance 下載台積電資料...")
# 下載台積電
ticker = "^GSPC"
df = yf.download(ticker, start="2016-01-01", end="2026-01-01")

df.columns = df.columns.droplevel('Ticker')
df.columns.name = None
df = df.reset_index()

df.rename(columns={"Date": "date"}, inplace=True)

df.to_csv("dataset/yfinance/GSPC-2016-2025a.csv", index=False)
