import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

def build_features(df):
    df['return'] = df['Close'].pct_change()
    df['ema9'] = EMAIndicator(df['Close'], 9).ema_indicator()
    df['ema21'] = EMAIndicator(df['Close'], 21).ema_indicator()
    df['rsi'] = RSIIndicator(df['Close'], 14).rsi()
    df['body'] = df['Close'] - df['Open']
    df['range'] = df['High'] - df['Low']
    return df.dropna()
