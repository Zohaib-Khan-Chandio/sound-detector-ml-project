"""Model loading and prediction (no Streamlit code here, so it is easy to test)."""
import base64
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import features_v2  # noqa: E402  (the exact feature code used for training)


def load_bundle(folder=HERE):
    paths = {n: os.path.join(folder, n) for n in ("model.pkl", "classes.json", "config.json")}
    missing = [n for n, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing file(s): " + ", ".join(missing))

    model = joblib.load(paths["model.pkl"])
    classes = json.load(open(paths["classes.json"]))
    config = json.load(open(paths["config.json"]))

    if not hasattr(model, "predict_proba") or type(model).__name__ == "SVC":
        raise ValueError("model.pkl must be a Pipeline/Stacking model with predict_proba (the one saved by Part 4).")
    n_feat = len(features_v2.FEATURE_COLUMNS)
    if hasattr(model, "n_features_in_") and model.n_features_in_ != n_feat:
        raise ValueError(f"Model expects {model.n_features_in_} features, features_v2.py makes {n_feat}.")

    class_ids = [int(c) for c in getattr(model, "classes_", range(len(classes)))]
    labels = [classes[i] for i in class_ids]

    # warm up librosa/numba so the first real request is not slow
    features_v2.extract_features_v2((np.random.randn(features_v2.SR * 2) * 0.05).astype("float32"))

    return {
        "model": model,
        "labels": labels,
        "threshold": float(config.get("threshold", 0.55)),
        "name": config.get("model_name", type(model).__name__),
    }


def predict_b64(bundle, pcm_b64):
    """pcm_b64: base64 of little-endian float32 samples, mono, 22050 Hz (first 4 s), made by the page."""
    try:
        raw = base64.b64decode(pcm_b64)
        if len(raw) < 4 or len(raw) % 4:
            raise ValueError
        y = np.frombuffer(raw, dtype="<f4").astype("float32")
    except Exception:
        raise ValueError("Could not read the audio sent by the page.")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) < 200:
        raise ValueError("Audio is too short.")

    feats = features_v2.extract_features_v2(y)
    if feats is None:
        raise ValueError("Feature extraction failed for this audio.")

    proba = bundle["model"].predict_proba(feats.reshape(1, -1))[0]
    labels = bundle["labels"]
    order = np.argsort(proba)[::-1]
    top3 = [{"label": labels[i], "prob": float(proba[i])} for i in order[:3]]
    return {
        "top3": top3,
        "label": top3[0]["label"],
        "confidence": top3[0]["prob"],
        "accepted": top3[0]["prob"] >= bundle["threshold"],
        "threshold": bundle["threshold"],
    }
