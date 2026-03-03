import yfinance as yf
import pandas as pd
import numpy as np
import os

print("1. 開始從 yfinance 下載台積電資料...")
# 下載台積電
ticker = "2330.TW"
df = yf.download(ticker, start="2016-01-01", end="2026-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()
df = df[["Date", "Open", "High", "Low", "Volume", "Close"]]

df.to_csv("dataset/yfinance/TSMC.csv", index=False)
