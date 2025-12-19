import os
import pandas as pd
import numpy as np
import joblib
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# =====================================================
# CONFIGURATION
# =====================================================
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_BASE = "backtest"
SCREENSHOT_COUNT = 5
TH = 0.60
FIXED_LOT = 0.01

# --- SETTING BALANCE ---
INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.01  # 1% risiko per trade
PIP_VALUE = 10         # Standard lot pip value (est)

os.makedirs(OUTPUT_BASE, exist_ok=True)

def get_pair_params(filename):
    if "XAUUSD" in filename.upper():
        return 0.01, 1, 100, 150
    return 0.00001, 10000, 10, 15

# Mencari file CSV di semua sub-folder
csv_files_full_path = []
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.csv'):
            csv_files_full_path.append(os.path.join(root, f))

print(f"DEBUG: Menemukan {len(csv_files_full_path)} file CSV.")

for full_path in csv_files_full_path:
    file_name = os.path.basename(full_path)
    pair_name = file_name.split('_')[0]
    POINT, PIP, SL_PIPS, TP_PIPS = get_pair_params(file_name)
    
    print(f"\n>>> Memproses {pair_name} dari: {full_path}")
    
    # 1. LOAD & CLEAN
    try:
        df = pd.read_csv(full_path, sep="\t")
        if "<OPEN>" not in df.columns and "Open" not in df.columns:
            df = pd.read_csv(full_path, sep=",")
        df.columns = [c.replace("<", "").replace(">", "").capitalize() for c in df.columns]
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.set_index('Datetime', inplace=True)
    except Exception as e:
        print(f"   - Gagal memuat data {file_name}: {e}")
        continue
    
    # 2. FEATURES
    df['return'] = df['Close'].pct_change()
    df['ema9'] = EMAIndicator(df['Close'], 9).ema_indicator()
    df['ema21'] = EMAIndicator(df['Close'], 21).ema_indicator()
    df['rsi'] = RSIIndicator(df['Close'], 14).rsi()
    df['body'] = df['Close'] - df['Open']
    df['range'] = df['High'] - df['Low']
    df.dropna(inplace=True)

    # 3. LOAD MODEL
    model_path = os.path.join(MODEL_DIR, f"{pair_name}.pkl")
    if not os.path.exists(model_path):
        print(f"   - SKIP: Model {model_path} tidak ditemukan!")
        continue
    model = joblib.load(model_path)
    
    X = df[['return', 'ema9', 'ema21', 'rsi', 'body', 'range']]
    proba = model.predict_proba(X)
    df['prob_buy'], df['prob_sell'] = proba[:, 1], proba[:, 0]
    df['signal'] = 'NO_TRADE'
    df.loc[df['prob_buy'] > TH, 'signal'] = 'BUY'
    df.loc[df['prob_sell'] > TH, 'signal'] = 'SELL'

    # 4. FAST BACKTEST
    df['result_pips'] = 0.0
    df['exit_type'] = 'NONE'
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['exit_index_val'] = np.nan

    sig_indices = np.where(df['signal'] != 'NO_TRADE')[0]
    high_arr, low_arr, open_arr, spread_arr = df['High'].values, df['Low'].values, df['Open'].values, df['Spread'].values

    for i in sig_indices:
        if i >= len(df) - 1: continue
        entry_idx = i + 1
        sig = df.iloc[i]['signal']
        spread = spread_arr[entry_idx] * POINT
        
        if sig == 'BUY':
            entry_p = open_arr[entry_idx] + spread
            sl_p, tp_p = entry_p - SL_PIPS/PIP, entry_p + TP_PIPS/PIP
            f_low, f_high = low_arr[entry_idx:], high_arr[entry_idx:]
            sl_hit, tp_hit = np.where(f_low <= sl_p)[0], np.where(f_high >= tp_p)[0]
        else:
            entry_p = open_arr[entry_idx]
            sl_p, tp_p = entry_p + SL_PIPS/PIP, entry_p - TP_PIPS/PIP
            f_high, f_low = high_arr[entry_idx:], low_arr[entry_idx:]
            sl_hit, tp_hit = np.where(f_high >= sl_p)[0], np.where(f_low <= tp_p)[0]

        f_sl = sl_hit[0] if len(sl_hit) > 0 else 999999
        f_tp = tp_hit[0] if len(tp_hit) > 0 else 999999
        if f_sl == 999999 and f_tp == 999999: continue
        
        if f_sl < f_tp:
            res, etype, eidx, ex_p = -SL_PIPS, 'SL', entry_idx + f_sl, sl_p
        else:
            res, etype, eidx, ex_p = TP_PIPS, 'TP', entry_idx + f_tp, tp_p

        idx_label = df.index[i]
        df.at[idx_label, 'result_pips'], df.at[idx_label, 'exit_type'] = res, etype
        df.at[idx_label, 'entry_price'], df.at[idx_label, 'exit_price'] = entry_p, ex_p
        df.at[idx_label, 'exit_index_val'] = eidx

    # 5. SUMMARY & BALANCE SIMULATION PER PAIR
    trades = df[df['exit_type'] != 'NONE'].copy()
    
    print(f"\n=== RESULT FOR {pair_name} ===")
    if len(trades) > 0:
        current_balance = INITIAL_BALANCE
        equity_curve = []
        for _, row in trades.iterrows():
            #risk_usd = current_balance * RISK_PER_TRADE
            #lot = risk_usd / (SL_PIPS * PIP_VALUE)
            lot = FIXED_LOT
            profit = row['result_pips'] * PIP_VALUE * lot
            current_balance += profit
            equity_curve.append(current_balance)
        
        equity = np.array(equity_curve)
        peak = equity[0]
        drawdowns = []

        for value in equity:
            if value > peak:
                peak = value
            drawdowns.append((peak - value) / peak)

        print(f"Total Trades    : {len(trades)}")
        print(f"Total Pips      : {round(trades['result_pips'].sum(), 2)}")
        print(f"Winrate         : {round((trades['result_pips'] > 0).mean() * 100, 2)} %")
        
        print("\n=== BALANCE SIMULATION ===")
        print("Initial Balance :", INITIAL_BALANCE)
        print("Final Balance   :", round(current_balance, 2))
        print("Net Profit      :", round(current_balance - INITIAL_BALANCE, 2))
        print("Max Drawdown    :", round(max(drawdowns) * 100, 2), "%")
        print("Return %        :", round((current_balance / INITIAL_BALANCE - 1) * 100, 2), "%")
    else:
        print("No trades found.")

    # 6. RANDOM SCREENSHOTS
    if len(trades) > 0:
        pair_out_dir = os.path.join(OUTPUT_BASE, pair_name)
        os.makedirs(pair_out_dir, exist_ok=True)
        samples = trades.sample(min(len(trades), SCREENSHOT_COUNT))
        for tid, (idx, row) in enumerate(samples.iterrows(), 1):
            entry_i, exit_i = df.index.get_loc(idx) + 1, int(row['exit_index_val'])
            start, end = max(0, entry_i - 10), min(len(df), exit_i + 15)
            chart_df = df.iloc[start:end][['Open','High','Low','Close']].copy()
            rel_entry, rel_exit = entry_i - start, exit_i - start
            
            buy_m = np.full(len(chart_df), np.nan); sell_m = np.full(len(chart_df), np.nan); exit_m = np.full(len(chart_df), np.nan)
            if row['signal'] == 'BUY': buy_m[rel_entry] = row['entry_price']
            else: sell_m[rel_entry] = row['entry_price']
            exit_m[rel_exit] = row['exit_price']

            apds = [
                mpf.make_addplot(buy_m, type='scatter', marker='^', markersize=100, color='green'),
                mpf.make_addplot(sell_m, type='scatter', marker='v', markersize=100, color='red'),
                mpf.make_addplot(exit_m, type='scatter', marker='x', markersize=80, color='black')
            ]
            apds = [a for a in apds if not np.all(np.isnan(a['data']))]
            mpf.plot(chart_df, type='candle', style='yahoo', addplot=apds,
                     title=f"{pair_name} | {row['signal']} | {row['result_pips']} Pips",
                     savefig=os.path.join(pair_out_dir, f"trade_{tid}.png"),warn_too_much_data=10000)