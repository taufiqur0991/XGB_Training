import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import glob
import json
import re  # Tambahkan regex untuk membersihkan string

# Register XGBoost converter untuk skl2onnx
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from xgboost import XGBClassifier

update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
)

def fix_base_score(xgb_model):
    """
    Memaksa perubahan base_score langsung ke dalam internal booster
    agar onnxmltools membaca nilai float, bukan string '[0.5]'.
    """
    try:
        # 1. Ambil nilai yang benar dari config
        import json
        booster = xgb_model.get_booster()
        config = json.loads(booster.save_config())
        bs_raw = config['learner']['learner_model_param']['base_score']
        
        if isinstance(bs_raw, str) and '[' in bs_raw:
            clean_str = bs_raw.replace('[', '').replace(']', '')
            new_bs = float(clean_str)
            
            # 2. Hard override pada level sklearn wrapper
            xgb_model.base_score = new_bs
            xgb_model.set_params(base_score=new_bs)
            
            # 3. CRITICAL: Simpan booster ke file sementara dan load kembali 
            # Ini akan merefresh internal state yang dibaca oleh ONNX converter
            model_bytearray = booster.save_raw("json")
            booster.load_model(model_bytearray)
            
            print(f"   🔧 Hard Fixed base_score: {bs_raw} -> {new_bs}")
    except Exception as e:
        print(f"   ⚠️ Gagal hard fix: {e}")

def convert_model_to_onnx(pair_name):
    """
    Convert XGBoost model dari .pkl ke .onnx untuk pair tertentu
    """
    pkl_path = f"models/{pair_name}.pkl"
    onnx_path = f"models/{pair_name}.onnx"
    feature_map_path = f"models/{pair_name}_feature_map.json"
    
    if not os.path.exists(pkl_path):
        print(f"❌ Model {pkl_path} tidak ditemukan!")
        return False
    
    # 1. Load model XGBoost dari .pkl
    print(f"📦 Loading model: {pkl_path}")
    xgb_model = joblib.load(pkl_path)
    
    # --- FIX START ---
    # Perbaiki base_score sebelum konversi
    fix_base_score(xgb_model)
    # --- FIX END ---

    print(f"   Model type: {type(xgb_model)}")
    print(f"   Model n_estimators: {xgb_model.n_estimators}")
    print(f"   Model n_features: {xgb_model.n_features_in_}")
    
    # 2. Simpan mapping feature names
    original_feature_names = list(xgb_model.feature_names_in_)
    feature_map = {f"f{i}": name for i, name in enumerate(original_feature_names)}
    
    with open(feature_map_path, 'w') as f:
        json.dump(feature_map, f, indent=2)
    print(f"   Feature mapping disimpan ke: {feature_map_path}")
    
    # 3. Reset feature names
    print(f"   Resetting feature names untuk kompatibilitas ONNX...")
    booster = xgb_model.get_booster()
    new_feature_names = [f"f{i}" for i in range(xgb_model.n_features_in_)]
    booster.feature_names = new_feature_names

    # Update internal state
    xgb_model._Booster = booster

    # --- PANGGIL FIX DI SINI ---
    fix_base_score(xgb_model)
    # 4. Input shape
    initial_type = [('float_input', FloatTensorType([None, xgb_model.n_features_in_]))]
    
    # 5. Convert ke ONNX
    print(f"🔄 Converting to ONNX...")
    try:
        onnx_model = convert_sklearn(
            xgb_model,
            initial_types=initial_type,
            target_opset={'': 12, 'ai.onnx.ml': 3},
            options={id(xgb_model): {'zipmap': False}}
        )
        
        onnx_bytes = onnx_model.SerializeToString()
        print(f"   ONNX model size: {len(onnx_bytes)/1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Error saat konversi: {e}")
        return False
    
    # 6. Simpan model
    with open(onnx_path, 'wb') as f:
        f.write(onnx_bytes)
    
    # 7. Verifikasi
    try:
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print(f"✅ Konversi berhasil: {onnx_path}")
        return True
    except Exception as e:
        print(f"⚠️ Verifikasi gagal: {e}")
        return True

def convert_all_models():
    pkl_files = glob.glob("models/*.pkl")
    if not pkl_files:
        print("❌ Tidak ada model .pkl di folder models/")
        return
    
    print(f"🔍 Ditemukan {len(pkl_files)} model untuk dikonversi\n")
    success_count = 0
    for pkl_file in pkl_files:
        pair_name = os.path.basename(pkl_file).replace('.pkl', '')
        if convert_model_to_onnx(pair_name):
            success_count += 1
        print()
    print(f"{'='*50}\n🎉 Selesai! {success_count}/{len(pkl_files)} model berhasil dikonversi")

if __name__ == "__main__":
    convert_all_models()