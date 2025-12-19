## 🧠 Training Generate models

### 📦 Dependency

```bash
pip install pandas numpy ta xgboost scikit-learn joblib mplfinance matplotlib
```
```bash
pip install onnx==1.15.0 onnxconverter-common==1.13.0 skl2onnx==1.15.0 onnxmltools==1.11.2
```

### ▶️ Run Training All Pair

```bash
python train_all.py
```


### ▶️ Run Training Specific Pair

```bash
python train_pair.py EURUSD
```

### ▶️ Run Backtest

```bash
python backtest_all.py
```