import numpy as np
import librosa
import warnings
from scipy.stats import skew, kurtosis

SR = 22050
MAX_LEN = int(SR * 4.0)
MIN_LEN = int(SR * 0.25)
TARGET_PEAK = 0.9
SILENCE_EPS = 1e-6
N_MFCC, N_MELS = 20, 32

STAT_FUNCS = {
    "mean":    lambda m: np.mean(m, axis=1),
    "std":     lambda m: np.std(m, axis=1),
    "min":     lambda m: np.min(m, axis=1),
    "max":     lambda m: np.max(m, axis=1),
    "median":  lambda m: np.median(m, axis=1),
    "skew":    lambda m: skew(m, axis=1),
    "kurt":    lambda m: kurtosis(m, axis=1),
    "absmean": lambda m: np.mean(np.abs(m), axis=1),
}
SINGLE_FEATS = ["centroid", "bandwidth", "rolloff", "flatness", "zcr"]
MFCC_STATS = ["mean", "std", "min", "max", "median", "skew", "kurt"]

def _names(prefix, n, stats):
    return [f"{prefix}{i:02d}_{s}" for s in stats for i in range(1, n + 1)]

FEATURE_COLUMNS = (
    _names("mfcc", N_MFCC, MFCC_STATS)
  + _names("delta", N_MFCC, ["absmean", "std"])
  + _names("delta2_", N_MFCC, ["absmean", "std"])
  + _names("mel", N_MELS, ["mean", "std"])
  + _names("contrast", 7, ["mean", "std"])
  + [f"{n}_{s}" for n in SINGLE_FEATS for s in ["mean", "std", "max"]]
  + [f"rms_{s}" for s in ["mean", "std", "max"]]
)

def load_and_prepare(audio):
    """audio: file path (str) ya already-loaded mono array at 22050 Hz."""
    if isinstance(audio, np.ndarray):
        y = audio.astype(np.float32)
    else:
        y, _ = librosa.load(audio, sr=SR, mono=True)
    if len(y) == 0:
        raise ValueError("empty audio")

    y = y[:MAX_LEN]
    if len(y) < MIN_LEN:
        reps = int(np.ceil(MIN_LEN / len(y)))
        y = np.tile(y, reps)[:MIN_LEN]

    rms_orig = librosa.feature.rms(y=y)[0]       # normalize se PEHLE

    peak = np.max(np.abs(y))
    y_norm = y / peak * TARGET_PEAK if peak > SILENCE_EPS else y
    return y_norm, rms_orig

def _fill(feats, prefix, mat, stats):
    for s in stats:
        vals = STAT_FUNCS[s](mat)
        for i, v in enumerate(vals, 1):
            feats[f"{prefix}{i:02d}_{s}"] = float(v)

def extract_features_v2(audio):
    """Returns float32 array of len(FEATURE_COLUMNS), ya fail hone par None."""
    try:
        y, rms_orig = load_and_prepare(audio)
        feats = {}
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
            _fill(feats, "mfcc", mfcc, MFCC_STATS)
            _fill(feats, "delta", librosa.feature.delta(mfcc), ["absmean", "std"])
            _fill(feats, "delta2_", librosa.feature.delta(mfcc, order=2), ["absmean", "std"])

            mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS), ref=1.0)
            _fill(feats, "mel", mel_db, ["mean", "std"])
            _fill(feats, "contrast", librosa.feature.spectral_contrast(y=y, sr=SR), ["mean", "std"])

            singles = {
                "centroid":  librosa.feature.spectral_centroid(y=y, sr=SR),
                "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=SR),
                "rolloff":   librosa.feature.spectral_rolloff(y=y, sr=SR),
                "flatness":  librosa.feature.spectral_flatness(y=y),
                "zcr":       librosa.feature.zero_crossing_rate(y=y),
            }
            for name, m in singles.items():
                feats[f"{name}_mean"] = float(np.mean(m))
                feats[f"{name}_std"]  = float(np.std(m))
                feats[f"{name}_max"]  = float(np.max(m))

            feats["rms_mean"] = float(np.mean(rms_orig))
            feats["rms_std"]  = float(np.std(rms_orig))
            feats["rms_max"]  = float(np.max(rms_orig))

        out = np.array([feats[c] for c in FEATURE_COLUMNS], dtype=np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        return None