import numpy as np

def build_label(df, pip, target_pips):
    y = np.zeros(len(df))

    for i in range(len(df) - 1):
        future = df.iloc[i + 1]
        price  = df.iloc[i]['Close']

        if (future['High'] - price) * pip >= target_pips:
            y[i] = 1    # BUY
        elif (price - future['Low']) * pip >= target_pips:
            y[i] = -1   # SELL
        else:
            y[i] = 0    # NO TRADE

    return y
