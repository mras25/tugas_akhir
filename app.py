import os as _os

# === Batasi memory TensorFlow agar tidak OOM di server ===
_os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"   # Matikan log TF yang tidak perlu
_os.environ["CUDA_VISIBLE_DEVICES"]   = "-1"  # Paksa CPU only, tidak alokasi GPU memory
_os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"   # Matikan OneDNN
_os.environ["TF_USE_LEGACY_KERAS"]    = "1"   # Pakai Keras 2.x untuk model .h5 lama

import tensorflow as tf

# Batasi TensorFlow hanya pakai memory sesuai kebutuhan (tidak pre-alokasi semua RAM)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib, json, os
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tf_keras.models import load_model
from scipy.ndimage import uniform_filter1d
from flask import session
from sklearn.model_selection import train_test_split
import gc
from datetime import datetime, timedelta

# =========================
# GLOBAL FUNCTION (WAJIB)
# =========================
def ema(data, alpha=0.5):
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[i-1])
    return np.array(result)


def postprocess(data, last_actual, blend_steps=5):
    # Offset alignment: geser prediksi agar nilai pertama sambung ke nilai aktual terakhir
    offset = last_actual - data[0]
    data = data + offset

    # Blend 5 timestep pertama agar transisi halus dari nilai aktual
    for i in range(min(blend_steps, len(data))):
        weight = (blend_steps - i) / blend_steps
        data[i] = weight * last_actual + (1 - weight) * data[i]

    # Clip nilai minimum -2 (sesuai notebook)
    data = np.clip(data, -2, None)
    return data

def safe_filename(name):
    return (
        name.replace("/", "_")
            .replace("\\", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
    )

def preprocess_like_training(df_raw, features):
    df = df_raw[features].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = np.clip(df[col], low, high)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.loc[:, df.std() != 0]
    return df

def get_threshold(df, feature):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(df[[feature]])
    centers = sorted(kmeans.cluster_centers_.flatten())
    return centers[1]



app = Flask(__name__)
app.secret_key = "secret_ta_rob_banjir"

ADMIN_KEY = "BMKGstasiunpanjanglampungitera"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# =========================
# PERSISTENT DISK (Render)
# Semua data user disimpan di sini agar tidak hilang saat redeploy.
# Mount path harus sama dengan yang diset di Render Dashboard > Disks.
# =========================
PERSISTENT_DIR = os.environ.get(
    "PERSISTENT_DIR",
    os.path.join("/opt/render/project/src/persistent_data")
)
DATA_DIR = PERSISTENT_DIR  # alias agar mudah dibaca

print("🔄 Loading scaler...")
print("🔄 Loading features...")


def load_artifacts_by_location(location):
    location_folder = location.lower().replace(" ", "_")
    base_path = os.path.join(MODEL_DIR, location_folder)
    model_path = os.path.join(base_path, f"{location_folder}.h5")
    scaler_path = os.path.join(base_path, "scaler_minmax_laut.pkl")
    features_path = os.path.join(base_path, "features.json")
    if not os.path.exists(model_path):
        return None, None, None
    # Load scaler & features saja dulu (ringan)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        features = json.load(f)
    # Model di-load terakhir, langsung dipakai, lalu di-clear
    model = load_model(model_path, compile=False)
    return model, scaler, features


def clear_model(model):
    """Hapus model dari memory setelah selesai dipakai"""
    try:
        del model
        gc.collect()
        tf.keras.backend.clear_session()
    except Exception:
        pass


print("✅ Model, scaler, dan features berhasil dimuat")


# =========================
# INIT DIRECTORIES (di persistent disk)
# =========================
for _dir in ["datasets", "predictions", "forecasts", "evaluations"]:
    os.makedirs(os.path.join(DATA_DIR, _dir), exist_ok=True)

WINDOW = 48


def handle_outliers_winsorize(data):
    df = data.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = np.clip(df[col], low, high)
    return df

def create_sequences_multistep(data, window=48, horizon=48):
    X, y = [], []
    for i in range(len(data) - window - horizon):
        X.append(data[i:i+window])
        y.append(data[i+window:i+window+horizon])
    return np.array(X), np.array(y)

def align_features(df, required_features):
    df_aligned = df.copy()
    for f in required_features:
        if f not in df_aligned.columns:
            df_aligned[f] = 0.0
    df_aligned = df_aligned[required_features]
    return df_aligned


# =========================
# ROUTES
# =========================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analisis")
def analisis():
    return render_template("analisis.html")

@app.route("/prediksi")
def prediksi():
    return render_template("prediksi.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route("/admin/login", methods=["POST"])
def admin_login():
    password = request.json.get("password")
    if password == "BMKGstasiunpanjanglampungitera":
        session["is_admin"] = True
        return jsonify({"status": "ok"})
    return jsonify({"error": "Password salah"}), 401

@app.route("/admin/status")
def admin_status():
    return jsonify({"is_admin": session.get("is_admin", False)})

def check_admin():
    return session.get("is_admin", False)

@app.route("/admin/logout")
def logout():
    session.clear()
    return jsonify({"status": "logout"})


# =========================
# UPLOAD
# =========================
@app.route("/api/upload", methods=["POST"])
def upload():
    if not check_admin():
        return jsonify({"error": "Akses hanya untuk admin"}), 403

    file = request.files.get("file")
    location = request.form.get("location")

    if not file or not location:
        return jsonify({"error": "File dan lokasi wajib diisi"}), 400

    location = location.lower()
    model, scaler, features = load_artifacts_by_location(location)

    if model is None:
        return jsonify({"error": f"Model untuk lokasi {location} tidak ditemukan"}), 400

    df_raw = pd.read_csv(file, sep=";", decimal=",", encoding="utf-8")

    required_columns = ["Time(UTC/GMT)", "Hsig(m)", "Hmax(m)", "elev_m", "WindSpeed(knots)"]
    missing = [col for col in required_columns if col not in df_raw.columns]
    if missing:
        return jsonify({"error": f"Format salah! Kolom hilang: {missing}"}), 400

    print("==== DEBUG DATA ====")
    print("Kolom CSV:", df_raw.columns.tolist())
    print("Sample data:\n", df_raw.head())
    print("Features model:", features)
    print("====================")

    time_col = None
    for col in df_raw.columns:
        if any(x in col.lower() for x in ["time", "date", "tanggal", "datetime"]):
            time_col = col
            break

    df_numeric = df_raw[features].copy()
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.ffill().bfill()
    df_proc = handle_outliers_winsorize(df_numeric)
    df_proc = df_proc.replace([np.inf, -np.inf], np.nan)
    df_proc = df_proc.dropna()

    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns
    df_proc_numeric = df_proc[numeric_cols]

    # CLUSTERING
    elevm_values = df_proc[["elev_m"]].values
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(elevm_values)
    df_proc["Cluster"] = clusters

    UPLOAD_DIR = os.path.join(DATA_DIR, "datasets")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{location}.csv")

    if time_col:
        df_final = pd.concat([
            df_raw[[time_col]].iloc[df_proc.index].reset_index(drop=True),
            df_proc.reset_index(drop=True)
        ], axis=1)
    else:
        df_final = df_proc

    df_final.to_csv(file_path, index=False)

    # CLEAR SEMUA CACHE — saat data baru diupload, semua hasil lama dihapus
    # agar user tidak melihat data lama. Admin harus jalankan ulang prediksi.
    for folder in ["predictions", "forecasts", "evaluations"]:
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.startswith(location):
                    os.remove(os.path.join(folder_path, f))

    return jsonify({
        "status": "ok",
        "rows": len(df_proc),
        "location": location,
        "validation": {
            "missing_features": [],
            "ignored_features": []
        }
    })


# =========================
# FEATURES
# =========================
@app.route("/api/features")
def api_features():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400
    file_path = os.path.join(DATA_DIR, "datasets", f"{location}.csv")
    if not os.path.exists(file_path):
        return jsonify({"error": "Dataset tidak ditemukan"}), 400
    df = pd.read_csv(file_path)
    model, scaler, features = load_artifacts_by_location(location)
    return jsonify({"features": features})


# =========================
# EVALUATE — Admin only, simpan ke cache (menggantikan lama)
# =========================
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    if not check_admin():
        return jsonify({"error": "Akses hanya untuk admin"}), 403

    location = request.json.get("location")
    model, scaler, features = load_artifacts_by_location(location)

    df_raw = pd.read_csv(os.path.join(DATA_DIR, "datasets", f"{location}.csv"))
    df = preprocess_like_training(df_raw, features)
    data_scaled = scaler.transform(df.values).astype(np.float32)

    window = 48
    horizon = 48
    X, y = create_sequences_multistep(data_scaled, window, horizon)

    if len(X) == 0:
        return jsonify({"error": "Data tidak cukup"}), 400

    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    y_pred = model.predict(X_test)

    y_pred_inv = scaler.inverse_transform(
        y_pred.reshape(-1, len(features))
    ).reshape(y_pred.shape)

    y_true_inv = scaler.inverse_transform(
        y_test.reshape(-1, len(features))
    ).reshape(y_test.shape)

    elev_idx = features.index("elev_m")
    y_pred_elevm = y_pred_inv[:, :, elev_idx]
    y_true_elevm = y_true_inv[:, :, elev_idx]

    y_pred_elevm = np.array([uniform_filter1d(seq, size=5) for seq in y_pred_elevm])

    rmse = float(np.sqrt(mean_squared_error(y_true_elevm.flatten(), y_pred_elevm.flatten())))
    mae  = float(mean_absolute_error(y_true_elevm.flatten(), y_pred_elevm.flatten()))

    # Hitung threshold dari KMeans (sama seperti notebook) dan simpan ke hasil evaluasi
    threshold = float(get_threshold(df, "elev_m"))

    df_full = pd.read_csv(os.path.join(DATA_DIR, "datasets", f"{location}.csv"))
    cluster_table = []
    cluster_stats = {}
    if "Cluster" in df_full.columns and "elev_m" in df_full.columns:
        stats = df_full.groupby("Cluster")["elev_m"].agg(["mean", "min", "max", "count"]).round(3)
        cluster_stats = stats.to_dict()
        cluster_table = stats.reset_index().to_dict(orient="records")

    result = {
        "rmse": rmse,
        "mae": mae,
        "threshold": threshold,
        "cluster_stats": cluster_stats,
        "cluster_table": cluster_table
    }

    # Simpan / timpa cache evaluasi
    eval_dir = os.path.join(DATA_DIR, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"{location}.json"), "w") as f:
        json.dump(result, f)

    return jsonify(result)


# =========================
# EVALUATION RESULT — Baca dari cache (user & admin)
# =========================
@app.route("/api/evaluation-result")
def evaluation_result():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400

    eval_file = os.path.join(DATA_DIR, "evaluations", f"{location}.json")
    if not os.path.exists(eval_file):
        return jsonify({"error": "Evaluasi belum dijalankan admin"}), 400

    with open(eval_file) as f:
        data = json.load(f)

    return jsonify({
        "rmse": data.get("rmse", 0),
        "mae": data.get("mae", 0),
        "threshold": data.get("threshold", None),
        "cluster_stats": data.get("cluster_stats", {}),
        "cluster_table": data.get("cluster_table", [])
    })


# =========================
# CLUSTER VISUAL
# =========================
@app.route("/api/cluster-visual", methods=["GET"])
def cluster_visual():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400

    file_path = os.path.join(DATA_DIR, "datasets", f"{location}.csv")
    if not os.path.exists(file_path):
        return jsonify({"error": "Dataset tidak ditemukan"}), 400

    df = pd.read_csv(file_path)
    if "Cluster" not in df.columns:
        return jsonify({"error": "Cluster belum tersedia"}), 400

    cluster_stats = df.groupby("Cluster")["elev_m"].agg(["mean", "min", "max", "count"]).round(3)
    cluster_table = cluster_stats.reset_index()

    return jsonify({
        "cluster_stats": cluster_stats.to_dict(),
        "cluster_table": cluster_table.to_dict(orient="records")
    })


# =========================
# PRED VS ACTUAL — Admin jalankan & simpan; user baca dari cache
# =========================
@app.route("/api/pred-vs-actual", methods=["GET", "POST"])
def pred_vs_actual():
    location = request.args.get("location") or (request.json or {}).get("location")
    feature_name = request.args.get("feature") or (request.json or {}).get("feature", "elev_m")

    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400
    
    run_model = request.args.get("run") == "1"

    # =========================
    # CACHE CONTROL (SISIPAN)
    # =========================
    pred_file = os.path.join(DATA_DIR, "predictions", f"{location}_{safe_filename(feature_name)}.json")

    # fallback lama
    if not os.path.exists(pred_file):
        pred_file = os.path.join(DATA_DIR, "predictions", f"{location}.json")

    # kalau TIDAK run → hanya baca cache
    if not run_model:
        if os.path.exists(pred_file):
            with open(pred_file) as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"error": "Cache belum ada, jalankan prediksi terlebih dahulu"}), 400

    # --- ADMIN: Jalankan model lalu simpan ---
    model, scaler, features = load_artifacts_by_location(location)
    if model is None:
        return jsonify({"error": f"Model untuk lokasi {location} tidak ditemukan"}), 400

    if feature_name not in features:
        feature_name = "elev_m"

    df_raw = pd.read_csv(os.path.join(DATA_DIR, "datasets", f"{location}.csv"))

    # TIME PARSE
    time_cols = [c for c in df_raw.columns if any(x in c.lower() for x in ["time", "date", "tanggal", "datetime"])]
    time_col = time_cols[0] if time_cols else None
    if time_col:
        df_raw["datetime"] = pd.to_datetime(df_raw[time_col], dayfirst=True, errors="coerce")
        df_raw = df_raw.dropna(subset=["datetime"])

    df = preprocess_like_training(df_raw, features)
    df = align_features(df, features)
    df_indices = df.index.tolist()

    data_scaled = scaler.transform(df.values).astype(np.float32)

    window = 48
    horizon = 48
    X, y = create_sequences_multistep(data_scaled, window, horizon)

    if len(X) == 0:
        return jsonify({"time": [], "pred": [], "actual": []})

    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    y_pred = model.predict(X_test)

    y_pred_inv = scaler.inverse_transform(
        y_pred.reshape(-1, len(features))
    ).reshape(y_pred.shape)
    y_test_inv = scaler.inverse_transform(
        y_test.reshape(-1, len(features))
    ).reshape(y_test.shape)

    feat_idx = features.index(feature_name)
    pred_last   = y_pred_inv[-1][:, feat_idx]
    actual_last = y_test_inv[-1][:, feat_idx]

    # TIDAK pakai postprocess di sini — agar perbandingan prediksi vs aktual fair
    # postprocess hanya dipakai untuk forecast ke depan (future prediction)

    # =========================
    # TIME ALIGNMENT (FIX FINAL)
    # =========================
    time_values = []

    if time_col:
        # ambil index sequence terakhir di test
        last_seq_idx = len(X_test) - 1

        # mapping ke index df
        start_idx = split_idx + last_seq_idx + window

        time_values = df_raw.iloc[
            start_idx : start_idx + horizon
        ]["datetime"].dt.strftime("%d-%m-%Y %H:%M").tolist()

    if not time_values:
        time_values = [f"+{i+1}j" for i in range(horizon)]

    threshold = get_threshold(df, feature_name)

    result = {
        "feature": feature_name,
        "time": time_values,
        "pred": pred_last.tolist(),
        "actual": actual_last.tolist(),
        "threshold": float(threshold)
    }

    # Simpan per-lokasi per-fitur (timpa yang lama)
    pred_dir = os.path.join(DATA_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    safe_feat = safe_filename(feature_name)
    with open(os.path.join(pred_dir, f"{location}_{safe_feat}.json"), "w") as f:
        json.dump(result, f)
    # Simpan juga file per-lokasi untuk kompatibilitas
    with open(os.path.join(pred_dir, f"{location}.json"), "w") as f:
        json.dump(result, f)

    clear_model(model)
    return jsonify(result)


# =========================
# FORECAST — Admin jalankan & simpan; timestamps nyata
# =========================
@app.route("/api/forecast", methods=["POST"])
def forecast():
    if not session.get("is_admin"):
        return jsonify({"error": "Hanya admin"}), 403

    location = request.json.get("location")
    feature_name = request.json.get("feature")

    model, scaler, features = load_artifacts_by_location(location)

    df_raw = pd.read_csv(os.path.join(DATA_DIR, "datasets", f"{location}.csv"))

    # Baca waktu terakhir dari dataset untuk generate timestamp forecast
    time_cols = [c for c in df_raw.columns if any(x in c.lower() for x in ["time", "date", "tanggal", "datetime"])]
    time_col = time_cols[0] if time_cols else None
    last_datetime = None
    if time_col:
        try:
            parsed_times = pd.to_datetime(df_raw[time_col], dayfirst=True, errors="coerce").dropna()
            if len(parsed_times) > 0:
                last_datetime = parsed_times.iloc[-1]
        except Exception:
            pass

    df = preprocess_like_training(df_raw, features)
    data_scaled = scaler.transform(df.values).astype(np.float32)

    window = 48
    last_seq = data_scaled[-window:].reshape(1, window, len(features))
    future_pred = model.predict(last_seq)

    future_pred_inv = scaler.inverse_transform(
        future_pred.reshape(-1, len(features))
    ).reshape(future_pred.shape)

    if not feature_name:
        feature_name = "elev_m"

    idx = features.index(feature_name)
    future = future_pred_inv[0][:, idx]

    last_actual = df[feature_name].iloc[-1] if feature_name in df.columns else future[0]
    future = postprocess(future, last_actual)

    threshold = get_threshold(df, feature_name)
    status = (future >= threshold).astype(int)

    # Generate timestamp nyata untuk setiap langkah forecast (per jam)
    horizon = len(future)
    if last_datetime is not None:
        steps = [
            (last_datetime + timedelta(hours=i + 1)).strftime("%d-%m-%Y %H:%M")
            for i in range(horizon)
        ]
    else:
        steps = [f"+{i + 1}j" for i in range(horizon)]

    result = {
        "values": future.tolist(),
        "status": status.tolist(),
        "threshold": float(threshold),
        "steps": steps  # timestamp nyata
    }

    forecasts_dir = os.path.join(DATA_DIR, "forecasts")
    os.makedirs(forecasts_dir, exist_ok=True)
    safe_feat = safe_filename(feature_name)
    with open(os.path.join(forecasts_dir, f"{location}_{safe_feat}.json"), "w") as f:
        json.dump(result, f)

    clear_model(model)
    return jsonify(result)


# =========================
# FORECAST RESULT — Baca dari cache (user & admin)
# =========================
@app.route("/api/forecast-result")
def forecast_result():
    location = request.args.get("location")
    feature  = request.args.get("feature")

    if not location or not feature:
        return jsonify({"error": "Parameter tidak lengkap"}), 400

    safe_feature = safe_filename(feature)
    file_path = os.path.join(DATA_DIR, "forecasts", f"{location}_{safe_feature}.json")

    if not os.path.exists(file_path):
        return jsonify({"error": "Forecast belum dijalankan admin"}), 400

    with open(file_path) as f:
        return jsonify(json.load(f))


# =========================
# PREDICTION RESULT — Baca dari cache (kompatibilitas lama)
# =========================
@app.route("/api/prediction-result")
def prediction_result():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400

    pred_file = os.path.join(DATA_DIR, "predictions", f"{location}.json")
    if not os.path.exists(pred_file):
        return jsonify({"error": "Prediksi belum dijalankan admin"}), 400

    with open(pred_file) as f:
        data = json.load(f)
    return jsonify(data)


# =========================
# TIMESERIES
# =========================
@app.route("/api/timeseries")
def api_timeseries():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400

    file_path = os.path.join(DATA_DIR, "datasets", f"{location}.csv")
    if not os.path.exists(file_path):
        return jsonify({"error": "Dataset tidak ditemukan"}), 400

    df = pd.read_csv(file_path)

    feature = request.args.get("feature")
    if not feature or feature not in df.columns:
        return jsonify({"error": "Fitur tidak valid"}), 400

    return jsonify({
        "location": location,
        "feature": feature,
        "values": df[feature].tolist(),
        "time": df["Time(UTC/GMT)"].tolist(),
    })


# =========================
# LOCATIONS
# =========================
@app.route("/api/locations")
def api_locations():
    datasets_dir = os.path.join(DATA_DIR, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    files = [f for f in os.listdir(datasets_dir) if f.endswith(".csv")]
    locations = [f.replace(".csv", "") for f in files]
    return jsonify({"locations": locations})


# =========================
# DASHBOARD
# =========================
@app.route("/api/dashboard")
def dashboard_data():
    location = request.args.get("location")
    model, scaler, features = load_artifacts_by_location(location)

    df = pd.read_csv(os.path.join(DATA_DIR, "datasets", f"{location}.csv"))
    df_proc = df[features].copy()
    df_proc = handle_outliers_winsorize(df_proc)
    df_proc = df_proc.replace([np.inf, -np.inf], np.nan).dropna()

    data_scaled = scaler.transform(df_proc.values).astype(np.float32)
    last_seq = data_scaled[-48:].reshape(1, 48, len(features))
    pred = model.predict(last_seq)

    pred_inv = scaler.inverse_transform(
        pred.reshape(-1, len(features))
    ).reshape(pred.shape)

    elev_idx = features.index("elev_m")
    future_elevm = pred_inv[0][:, elev_idx]

    # === Sesuai notebook: offset alignment → blend → clip (tanpa EMA) ===
    last_actual = df_proc["elev_m"].iloc[-1]
    future_elevm = postprocess(future_elevm, last_actual)

    # KMeans di-fit pada df_proc (sudah bersih), sesuai notebook
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(df_proc[["elev_m"]])
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = centers[1]  # centroid tertinggi = threshold banjir rob

    flood_idx = np.where(future_elevm >= threshold)[0]

    # Cari waktu nyata potensi banjir pertama
    flood_time = None
    if len(flood_idx) > 0:
        jam_ke = int(flood_idx[0] + 1)
        waktu_banjir = datetime.now() + timedelta(hours=jam_ke)
        flood_time = waktu_banjir.strftime("%d-%m-%Y %H:%M")

    return jsonify({
        "status": "BANJIR" if len(flood_idx) > 0 else "AMAN",
        "elevm": float(future_elevm[-1]),
        "threshold": float(threshold),
        "flood_time": flood_time  # contoh: "03-05-2025 14:00"
    })


if __name__ == "__main__":
    print("🚀 START FLASK SERVER")
    app.run(debug=True, host="127.0.0.1", port=5000)
