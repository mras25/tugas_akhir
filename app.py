from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib, json, os
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import uniform_filter1d
from datetime import datetime, timedelta

# ===========================
# LAZY IMPORT KERAS (hemat RAM saat startup)
# ===========================
_load_model = None
def get_keras_load_model():
    global _load_model
    if _load_model is None:
        from keras.models import load_model
        _load_model = load_model
    return _load_model


# =========================
# GLOBAL FUNCTIONS
# =========================
def ema(data, alpha=0.5):
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[i-1])
    return np.array(result)


def postprocess(data, last_actual, blend_steps=5):
    offset = last_actual - data[0]
    data = data + offset
    for i in range(min(blend_steps, len(data))):
        weight = (blend_steps - i) / blend_steps
        data[i] = weight * last_actual + (1 - weight) * data[i]
    return np.clip(data, -2, None)


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
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[:, df.std() != 0]
    return df


def get_threshold(df, feature):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(df[[feature]])
    centers = sorted(kmeans.cluster_centers_.flatten())
    return centers[1]


# =========================
# MODEL CACHE (in-memory, per lokasi)
# =========================
_model_cache = {}

def load_artifacts_by_location(location):
    if location in _model_cache:
        return _model_cache[location]

    load_model = get_keras_load_model()
    location_folder = location.lower().replace(" ", "_")
    base_path = os.path.join(MODEL_DIR, location_folder)
    model_path = os.path.join(base_path, f"{location_folder}.h5")
    scaler_path = os.path.join(base_path, "scaler_minmax_laut.pkl")
    features_path = os.path.join(base_path, "features.json")

    if not os.path.exists(model_path):
        return None, None, None

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        features = json.load(f)

    _model_cache[location] = (model, scaler, features)
    return model, scaler, features


# =========================
# APP INIT
# =========================
app = Flask(__name__)
app.secret_key = "secret_ta_rob_banjir"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

for _dir in ["datasets", "predictions", "forecasts", "evaluations"]:
    os.makedirs(os.path.join(BASE_DIR, _dir), exist_ok=True)

WINDOW = 48


# =========================
# HELPERS
# =========================
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
    return df_aligned[required_features]


def check_admin():
    return session.get("is_admin", False)


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

    time_col = next(
        (col for col in df_raw.columns
         if any(x in col.lower() for x in ["time", "date", "tanggal", "datetime"])),
        None
    )

    df_numeric = df_raw[features].copy()
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df_proc = handle_outliers_winsorize(df_numeric).replace([np.inf, -np.inf], np.nan).dropna()

    clusters = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(df_proc[["elev_m"]].values)
    df_proc["Cluster"] = clusters

    file_path = os.path.join(BASE_DIR, "datasets", f"{location}.csv")
    if time_col:
        df_final = pd.concat([
            df_raw[[time_col]].iloc[df_proc.index].reset_index(drop=True),
            df_proc.reset_index(drop=True)
        ], axis=1)
    else:
        df_final = df_proc

    df_final.to_csv(file_path, index=False)

    for folder in ["predictions", "forecasts", "evaluations"]:
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.startswith(location):
                    os.remove(os.path.join(folder_path, f))

    return jsonify({
        "status": "ok",
        "rows": len(df_proc),
        "location": location,
        "validation": {"missing_features": [], "ignored_features": []}
    })


# =========================
# FEATURES
# =========================
@app.route("/api/features")
def api_features():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400
    if not os.path.exists(os.path.join(BASE_DIR, "datasets", f"{location}.csv")):
        return jsonify({"error": "Dataset tidak ditemukan"}), 400
    _, _, features = load_artifacts_by_location(location)
    return jsonify({"features": features})


# =========================
# EVALUATE
# =========================
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    if not check_admin():
        return jsonify({"error": "Akses hanya untuk admin"}), 403

    location = request.json.get("location")
    model, scaler, features = load_artifacts_by_location(location)

    df_raw = pd.read_csv(os.path.join(BASE_DIR, "datasets", f"{location}.csv"))
    df = preprocess_like_training(df_raw, features)
    data_scaled = scaler.transform(df).astype(np.float32)

    window, horizon = 48, 48
    X, y = create_sequences_multistep(data_scaled, window, horizon)
    if len(X) == 0:
        return jsonify({"error": "Data tidak cukup"}), 400

    split_idx = int(len(X) * 0.8)
    y_pred = model.predict(X[split_idx:])
    y_test = y[split_idx:]

    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, len(features))).reshape(arr.shape)

    elev_idx = features.index("elev_m")
    y_pred_elevm = np.array([uniform_filter1d(seq, size=5) for seq in inv(y_pred)[:, :, elev_idx]])
    y_true_elevm = inv(y_test)[:, :, elev_idx]

    rmse = float(np.sqrt(mean_squared_error(y_true_elevm.flatten(), y_pred_elevm.flatten())))
    mae  = float(mean_absolute_error(y_true_elevm.flatten(), y_pred_elevm.flatten()))
    threshold = float(get_threshold(df, "elev_m"))

    df_full = pd.read_csv(os.path.join(BASE_DIR, "datasets", f"{location}.csv"))
    cluster_table, cluster_stats = [], {}
    if "Cluster" in df_full.columns and "elev_m" in df_full.columns:
        stats = df_full.groupby("Cluster")["elev_m"].agg(["mean", "min", "max", "count"]).round(3)
        cluster_stats = stats.to_dict()
        cluster_table = stats.reset_index().to_dict(orient="records")

    result = {"rmse": rmse, "mae": mae, "threshold": threshold,
              "cluster_stats": cluster_stats, "cluster_table": cluster_table}

    eval_dir = os.path.join(BASE_DIR, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"{location}.json"), "w") as f:
        json.dump(result, f)

    return jsonify(result)


# =========================
# EVALUATION RESULT
# =========================
@app.route("/api/evaluation-result")
def evaluation_result():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400
    eval_file = os.path.join(BASE_DIR, "evaluations", f"{location}.json")
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
@app.route("/api/cluster-visual")
def cluster_visual():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400
    file_path = os.path.join(BASE_DIR, "datasets", f"{location}.csv")
    if not os.path.exists(file_path):
        return jsonify({"error": "Dataset tidak ditemukan"}), 400
    df = pd.read_csv(file_path)
    if "Cluster" not in df.columns:
        return jsonify({"error": "Cluster belum tersedia"}), 400
    stats = df.groupby("Cluster")["elev_m"].agg(["mean", "min", "max", "count"]).round(3)
    return jsonify({
        "cluster_stats": stats.to_dict(),
        "cluster_table": stats.reset_index().to_dict(orient="records")
    })


# =========================
# PRED VS ACTUAL
# =========================
@app.route("/api/pred-vs-actual", methods=["GET", "POST"])
def pred_vs_actual():
    location = request.args.get("location") or (request.json or {}).get("location")
    feature_name = request.args.get("feature") or (request.json or {}).get("feature", "elev_m")

    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400

    run_model = request.args.get("run") == "1"

    safe_feat = safe_filename(feature_name)
    pred_file = os.path.join(BASE_DIR, "predictions", f"{location}_{safe_feat}.json")
    if not os.path.exists(pred_file):
        pred_file = os.path.join(BASE_DIR, "predictions", f"{location}.json")

    if not run_model:
        if os.path.exists(pred_file):
            with open(pred_file) as f:
                return jsonify(json.load(f))
        return jsonify({"error": "Cache belum ada, jalankan prediksi terlebih dahulu"}), 400

    model, scaler, features = load_artifacts_by_location(location)
    if model is None:
        return jsonify({"error": f"Model untuk lokasi {location} tidak ditemukan"}), 400

    if feature_name not in features:
        feature_name = "elev_m"

    df_raw = pd.read_csv(os.path.join(BASE_DIR, "datasets", f"{location}.csv"))

    time_cols = [c for c in df_raw.columns if any(x in c.lower() for x in ["time", "date", "tanggal", "datetime"])]
    time_col = time_cols[0] if time_cols else None
    if time_col:
        df_raw["datetime"] = pd.to_datetime(df_raw[time_col], errors="coerce")
        df_raw = df_raw.dropna(subset=["datetime"])

    df = align_features(preprocess_like_training(df_raw, features), features)
    data_scaled = scaler.transform(df).astype(np.float32)

    window, horizon = 48, 48
    X, y = create_sequences_multistep(data_scaled, window, horizon)
    if len(X) == 0:
        return jsonify({"time": [], "pred": [], "actual": []})

    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]
    y_pred = model.predict(X_test)

    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, len(features))).reshape(arr.shape)

    feat_idx = features.index(feature_name)
    pred_last   = inv(y_pred)[-1][:, feat_idx]
    actual_last = inv(y_test)[-1][:, feat_idx]

    time_values = []
    if time_col:
        start_idx = split_idx + len(X_test) - 1 + window
        time_values = df_raw.iloc[start_idx:start_idx + horizon]["datetime"].dt.strftime("%d-%m-%Y %H:%M").tolist()
    if not time_values:
        time_values = [f"+{i+1}j" for i in range(horizon)]

    result = {
        "feature": feature_name,
        "time": time_values,
        "pred": pred_last.tolist(),
        "actual": actual_last.tolist(),
        "threshold": float(get_threshold(df, feature_name))
    }

    pred_dir = os.path.join(BASE_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    for fname in [f"{location}_{safe_filename(feature_name)}.json", f"{location}.json"]:
        with open(os.path.join(pred_dir, fname), "w") as f:
            json.dump(result, f)

    return jsonify(result)


# =========================
# FORECAST
# =========================
@app.route("/api/forecast", methods=["POST"])
def forecast():
    if not session.get("is_admin"):
        return jsonify({"error": "Hanya admin"}), 403

    location = request.json.get("location")
    feature_name = request.json.get("feature") or "elev_m"

    model, scaler, features = load_artifacts_by_location(location)
    df_raw = pd.read_csv(os.path.join(BASE_DIR, "datasets", f"{location}.csv"))

    time_cols = [c for c in df_raw.columns if any(x in c.lower() for x in ["time", "date", "tanggal", "datetime"])]
    last_datetime = None
    if time_cols:
        try:
            parsed = pd.to_datetime(df_raw[time_cols[0]], errors="coerce").dropna()
            if len(parsed) > 0:
                last_datetime = parsed.iloc[-1]
        except Exception:
            pass

    df = preprocess_like_training(df_raw, features)
    data_scaled = scaler.transform(df).astype(np.float32)

    future_pred = model.predict(data_scaled[-48:].reshape(1, 48, len(features)))
    future_pred_inv = scaler.inverse_transform(
        future_pred.reshape(-1, len(features))
    ).reshape(future_pred.shape)

    idx = features.index(feature_name)
    future = postprocess(future_pred_inv[0][:, idx],
                         df[feature_name].iloc[-1] if feature_name in df.columns else future_pred_inv[0][0, idx])

    threshold = get_threshold(df, feature_name)
    horizon = len(future)

    if last_datetime is not None:
        steps = [(last_datetime + timedelta(hours=i + 1)).strftime("%d-%m-%Y %H:%M") for i in range(horizon)]
    else:
        steps = [f"+{i+1}j" for i in range(horizon)]

    result = {
        "values": future.tolist(),
        "status": (future >= threshold).astype(int).tolist(),
        "threshold": float(threshold),
        "steps": steps
    }

    forecasts_dir = os.path.join(BASE_DIR, "forecasts")
    os.makedirs(forecasts_dir, exist_ok=True)
    with open(os.path.join(forecasts_dir, f"{location}_{safe_filename(feature_name)}.json"), "w") as f:
        json.dump(result, f)

    return jsonify(result)


# =========================
# FORECAST RESULT
# =========================
@app.route("/api/forecast-result")
def forecast_result():
    location = request.args.get("location")
    feature  = request.args.get("feature")
    if not location or not feature:
        return jsonify({"error": "Parameter tidak lengkap"}), 400
    file_path = os.path.join(BASE_DIR, "forecasts", f"{location}_{safe_filename(feature)}.json")
    if not os.path.exists(file_path):
        return jsonify({"error": "Forecast belum dijalankan admin"}), 400
    with open(file_path) as f:
        return jsonify(json.load(f))


# =========================
# PREDICTION RESULT
# =========================
@app.route("/api/prediction-result")
def prediction_result():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400
    pred_file = os.path.join(BASE_DIR, "predictions", f"{location}.json")
    if not os.path.exists(pred_file):
        return jsonify({"error": "Prediksi belum dijalankan admin"}), 400
    with open(pred_file) as f:
        return jsonify(json.load(f))


# =========================
# TIMESERIES
# =========================
@app.route("/api/timeseries")
def api_timeseries():
    location = request.args.get("location")
    if not location:
        return jsonify({"error": "Lokasi belum dipilih"}), 400
    file_path = os.path.join(BASE_DIR, "datasets", f"{location}.csv")
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
    datasets_dir = os.path.join(BASE_DIR, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    return jsonify({
        "locations": [f.replace(".csv", "") for f in os.listdir(datasets_dir) if f.endswith(".csv")]
    })


# =========================
# DASHBOARD — baca dari cache forecast (hemat RAM)
# =========================
@app.route("/api/dashboard")
def dashboard_data():
    location = request.args.get("location")

    # Prioritas: baca dari cache forecast agar tidak perlu jalankan model
    forecast_file = os.path.join(BASE_DIR, "forecasts", f"{location}_elev_m.json")
    if os.path.exists(forecast_file):
        with open(forecast_file) as f:
            data = json.load(f)
        future_elevm = np.array(data["values"])
        threshold = data["threshold"]
        flood_idx = np.where(future_elevm >= threshold)[0]
        flood_time = None
        if len(flood_idx) > 0:
            flood_time = (datetime.now() + timedelta(hours=int(flood_idx[0]) + 1)).strftime("%d-%m-%Y %H:%M")
        return jsonify({
            "status": "BANJIR" if len(flood_idx) > 0 else "AMAN",
            "elevm": float(future_elevm[-1]),
            "threshold": float(threshold),
            "flood_time": flood_time
        })

    # Fallback jika cache belum ada (pertama kali sebelum admin run forecast)
    model, scaler, features = load_artifacts_by_location(location)
    df = pd.read_csv(os.path.join(BASE_DIR, "datasets", f"{location}.csv"))
    df_proc = handle_outliers_winsorize(df[features].copy()).replace([np.inf, -np.inf], np.nan).dropna()

    data_scaled = scaler.transform(df_proc).astype(np.float32)
    pred = model.predict(data_scaled[-48:].reshape(1, 48, len(features)))
    pred_inv = scaler.inverse_transform(pred.reshape(-1, len(features))).reshape(pred.shape)

    elev_idx = features.index("elev_m")
    future_elevm = postprocess(pred_inv[0][:, elev_idx], df_proc["elev_m"].iloc[-1])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(df_proc[["elev_m"]])
    threshold = sorted(kmeans.cluster_centers_.flatten())[1]

    flood_idx = np.where(future_elevm >= threshold)[0]
    flood_time = None
    if len(flood_idx) > 0:
        flood_time = (datetime.now() + timedelta(hours=int(flood_idx[0]) + 1)).strftime("%d-%m-%Y %H:%M")

    return jsonify({
        "status": "BANJIR" if len(flood_idx) > 0 else "AMAN",
        "elevm": float(future_elevm[-1]),
        "threshold": float(threshold),
        "flood_time": flood_time
    })


if __name__ == "__main__":
    print("Mulai Flask server")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
