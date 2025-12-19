import glob, os
import pandas as pd
import joblib
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.metrics import accuracy_score

# Fungsi untuk memproses data persis seperti Kode 1 (Mode Menghafal)
def build_features_spesifik(df):
    # Feature Engineering dari Kode 1
    df['return'] = df['Close'].pct_change()
    df['ema9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['Close'], window=21).ema_indicator()
    df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
    df['body'] = df['Close'] - df['Open']
    df['range'] = df['High'] - df['Low']
    
    # Target dari Kode 1 (Mode Pasrah/Menghafal arah candle depan)
    # 1 jika naik, 0 jika turun
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

def train_pair(pair_name):
    print(f"\n🚀 Training Spesifik untuk: {pair_name}")

    # 1. Ambil semua file CSV di folder pair tersebut
    path = f"data/{pair_name}/*.csv"
    files = glob.glob(path)
    
    if not files:
        print(f"❌ Data untuk {pair_name} tidak ditemukan di {path}")
        return

    dfs = []
    for f in files:
        temp_df = pd.read_csv(f, sep="\t")
        temp_df.rename(columns={
            "<OPEN>": "Open", "<HIGH>": "High",
            "<LOW>": "Low", "<CLOSE>": "Close",
            "<TICKVOL>": "TickVolume"
        }, inplace=True, errors='ignore')
        dfs.append(temp_df)

    # Gabungkan semua data histori untuk pair ini
    df = pd.concat(dfs, ignore_index=True)
    
    # 2. Build Features & Target
    df = build_features_spesifik(df)
    
    X = df[['return', 'ema9', 'ema21', 'rsi', 'body', 'range']]
    y = df['target']

    # 3. Model XGBoost (Mode Binary seperti Kode 1)
    model = xgb.XGBClassifier(
        n_estimators=1000, # Tambah estimator agar lebih kuat menghafal
        max_depth=10,      # Lebih dalam agar lebih detail menghafalnya
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss'
    )

    # Train model hanya pada data pair ini
    model.fit(X, y)

    # 4. Simpan Model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{pair_name}.pkl"
    joblib.dump(model, model_path)
    
    # Cek Akurasi Backtest (In-Sample)
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    print(f"✅ Model {pair_name} saved. Memorization Accuracy: {acc:.2%}")