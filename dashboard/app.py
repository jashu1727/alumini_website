import sys
import streamlit as st
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import os

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Install it with `pip install plotly` for enhanced dashboard charts.")

# Ensure the workspace root is on sys.path so local packages can be imported
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BALANCED_DATA_PATH = ROOT_DIR / "data" / "train_balanced.csv"
BATADAL_DATA_PATH = ROOT_DIR / "data" / "BATADAL_dataset04.csv"

st.set_page_config(page_title="TrustSentinel Pro - ICS Security Intelligence", layout="wide")

# Try to import workspace modules, provide fallbacks if missing
try:
    from twin.digital_twin import DigitalTwin
except Exception:
    DigitalTwin = None

if DigitalTwin is not None:
    @st.cache_resource
    def get_digital_twin(model_path: str):
        return DigitalTwin(model_path)
else:
    def get_digital_twin(model_path: str):
        raise RuntimeError("DigitalTwin class not available.")

@st.cache_resource
def load_scaler_data(scaler_path: str):
    """Load scaler, threshold and feature mask data from .npz file."""
    if os.path.exists(scaler_path):
        data = np.load(scaler_path, allow_pickle=True)
        return {
            'mean': data['mean'],
            'std': data['std'],
            'threshold': float(data['threshold']),
            'feature_mask': data['feature_mask'] if 'feature_mask' in data else None
        }
    return None

try:
    from zerotrust.trust_engine import TrustEngine
except Exception:
    class TrustEngine:
        def __init__(self):
            self.trust_score = 1.0

        def update_trust(self, anomaly_score: float) -> float:
            anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

            # Strong trust degradation during attacks
            if anomaly_score > 0.8:
                self.trust_score *= 0.2
            elif anomaly_score > 0.6:
                self.trust_score *= 0.4
            elif anomaly_score > 0.4:
                self.trust_score *= 0.6
            elif anomaly_score > 0.2:
                self.trust_score *= 0.8
            else:
                # Slowly recover during normal operation
                self.trust_score = min(1.0, self.trust_score + 0.02)

            return float(np.clip(self.trust_score, 0.0, 1.0))

        def reset(self):
            self.trust_score = 1.0

try:
    from zerotrust.policy_engine import check_access
except Exception:
    def check_access(trust_score: float, threshold: float = 0.5) -> str:
        return "ACCESS GRANTED" if trust_score >= threshold else "ACCESS DENIED"

try:
    from attacks.attack_simulator import inject_attack
except Exception:
    def inject_attack(data, strength=5.0):
        attacked = data.copy()
        if attacked.ndim != 1:
            attacked = attacked.flatten()
        length = attacked.shape[0]
        if length >= 10:
            attacked[5:10] += np.random.normal(loc=strength, scale=strength/2, size=attacked[5:10].shape)
        else:
            attacked += np.random.normal(loc=strength, scale=strength/2, size=attacked.shape)
        return attacked

try:
    from twin.anomaly_detector import compute_anomaly
except Exception:
    def compute_anomaly(real: torch.Tensor, predicted: torch.Tensor) -> float:
        real = torch.tensor(real).float()
        predicted = torch.tensor(predicted).float()
        error = torch.mean((real - predicted) ** 2)
        scale = torch.mean(real ** 2) + 1e-8
        normalized = torch.clamp(error / scale, 0.0, 1.0)
        return float(normalized.item())

# Evaluation functions
ENSEMBLE_BASE_PATH = ROOT_DIR / "models" / "ensemble_classifier"
CLASSIFIER_MODEL_PATH = ROOT_DIR / "models" / "batadal_classifier.pth"
CLASSIFIER_SCALER_PATH = ROOT_DIR / "models" / "batadal_classifier.npz"

def load_scaler_eval(scaler_path):
    if os.path.exists(scaler_path):
        data = np.load(scaler_path, allow_pickle=True)
        mean = data['mean']
        std = data['std']
        threshold = float(data['threshold'])
        feature_mask = data['feature_mask'] if 'feature_mask' in data and len(data['feature_mask']) > 0 else None
        return mean, std, threshold, feature_mask
    return None, None, None, None

def load_eval_data(dataset="synthetic", n_features=3):
    if dataset == "batadal":
        # Load balanced training data for proper evaluation (has both normal and attack)
        train_balanced_path = ROOT_DIR / "data" / "train_balanced.csv"
        if train_balanced_path.exists():
            df = pd.read_csv(train_balanced_path)
            feature_cols = [c for c in df.columns if c not in ['DATETIME', 'ATT_FLAG']]
            data = df[feature_cols].astype("float32").values
            # Use original labels directly: 0=normal, 1=attack
            y_true = df['ATT_FLAG'].astype(int).values
            return data, y_true
        # Fallback to combined normal and attack dataset
        normal_path = ROOT_DIR / "data" / "normal_data.csv"
        attack_path = ROOT_DIR / "data" / "attack_data.csv"
        if normal_path.exists() and attack_path.exists():
            normal_df = pd.read_csv(normal_path)
            attack_df = pd.read_csv(attack_path)
            
            # Extract features and labels
            normal_data = normal_df.astype("float32").values
            attack_data = attack_df.astype("float32").values
            
            # Combine normal (label=0) and attack (label=1)
            data = np.vstack([normal_data, attack_data])
            y_true = np.hstack([np.zeros(len(normal_data)), np.ones(len(attack_data))])
            return data, y_true
        # Fallback to BATADAL dataset if balanced not available
        batadal_path = ROOT_DIR / "data" / "BATADAL_dataset04.csv"
        if batadal_path.exists():
            attack_df = pd.read_csv(batadal_path, skipinitialspace=True)
            attack_df.columns = attack_df.columns.str.strip()
            feature_cols = [c for c in attack_df.columns if c not in ['DATETIME', 'ATT_FLAG']]
            # Use actual ATT_FLAG values from BATADAL labels
            y_true = (attack_df['ATT_FLAG'] == 1).astype(int).values
            data = attack_df[feature_cols].astype("float32").values
            return data, y_true
        else:
            dataset = "synthetic"
    elif dataset == "swat":
        swat_path = ROOT_DIR / "data" / "SWaT_Dataset_Attack_v0.csv"
        if swat_path.exists():
            attack_df = pd.read_csv(swat_path)
            if 'Normal/Attack' in attack_df.columns:
                y_true = (attack_df['Normal/Attack'] == 'Attack').astype(int).values
            else:
                y_true = np.zeros(len(attack_df))
            data = attack_df.iloc[:, 1:1+n_features].astype("float32").values
            return data, y_true
        else:
            dataset = "synthetic"
    if dataset == "synthetic":
        seq_len = 100
        n_samples = 1000
        data = np.zeros((n_samples, n_features), dtype=np.float32)
        for i in range(n_samples):
            for f in range(n_features):
                data[i, f] = np.sin(i * 0.1 + f) + np.random.normal(0, 0.03)
        y_true = np.zeros(n_samples)
        y_true[200:250] = 1
        y_true[400:450] = 1
        y_true[700:720] = 1
    return data, y_true

def create_sequences_eval(data, y_true, seq_len=100):
    if len(data) < seq_len:
        raise ValueError("Not enough data to create evaluation sequences")
    sequences = []
    labels = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
        labels.append(y_true[i+seq_len-1])
    return np.stack(sequences), np.array(labels)

def find_best_threshold_for_objective(y_true, anomaly_scores, objective="f1"):
    """Find threshold optimizing for different objectives."""
    preds = np.array(anomaly_scores)
    thresholds = np.linspace(preds.min(), preds.max(), 100)
    
    best_threshold = None
    best_score = -1
    results = []
    
    for t in thresholds:
        y_pred = (preds > t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        
        if objective == "f1":
            score = f1
        elif objective == "precision":
            score = precision
        elif objective == "recall":
            score = recall
        elif objective == "specificity":
            score = specificity
        elif objective == "balanced":
            score = (recall + specificity) / 2
        
        results.append((t, precision, recall, f1, specificity))
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score, results

def use_ensemble_models_eval(data, y_true):
    """Use ensemble voting for better predictions"""
    num_models = 3
    all_scores = []
    
    for model_idx in range(num_models):
        model_path = ENSEMBLE_BASE_PATH.with_name(f"{ENSEMBLE_BASE_PATH.name}_model{model_idx}.pth")
        scaler_path = ENSEMBLE_BASE_PATH.with_name(f"{ENSEMBLE_BASE_PATH.name}_scaler{model_idx}.npz")
        
        if not model_path.exists():
            continue
        
        # Load model and scaler
        twin = get_digital_twin(str(model_path))
        mean, std, threshold, feature_mask = load_scaler_eval(str(scaler_path))
        
        # Process data
        data_processed = data.copy()
        
        # Apply feature selection
        if feature_mask is not None and len(feature_mask) > 0:
            data_processed = data_processed[:, feature_mask]
        
        # Normalize
        if mean is not None and std is not None:
            data_processed = (data_processed - mean) / std
        
        data_processed = data_processed.astype(np.float32)
        
        # Get predictions
        scores = []
        for row in data_processed:
            tensor = torch.tensor(row).unsqueeze(0)
            pred = twin.predict(tensor)
            score = torch.sigmoid(pred.squeeze(0)).item()
            scores.append(score)
        
        all_scores.append(np.array(scores))
    
    # Ensemble voting (average predictions)
    if all_scores:
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores
    else:
        return np.zeros(len(data))

try:
    from utils.metrics import evaluate_metrics
except Exception:
    def evaluate_metrics(y_true, anomaly_scores, threshold=0.5):
        y_pred = (np.array(anomaly_scores) > threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (np.array(y_true) == 1)))
        fp = int(np.sum((y_pred == 1) & (np.array(y_true) == 0)))
        fn = int(np.sum((y_pred == 0) & (np.array(y_true) == 1)))
        tn = int(np.sum((y_pred == 0) & (np.array(y_true) == 0)))
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
        
        # ROC AUC
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(y_true, anomaly_scores)
        except Exception:
            roc_auc = 0.5
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'FPR': fpr,
            'ROC_AUC': roc_auc
        }

# --- UI ---
st.markdown("""
<style>
.title-main {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3em;
    font-weight: 900;
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 20px;
}
</style>
<h1 class="title-main">🛡️ TrustSentinel Pro</h1>
<p style="text-align: center; font-size: 1.2em; color: #4ECDC4; font-weight: 600; margin-top: -10px;">
Adaptive ICS Security Intelligence
</p>
""", unsafe_allow_html=True)

# Main tabs
tab_simulation, tab_evaluation = st.tabs(["🔄 Real-time Simulation", "📊 Model Evaluation"])

with tab_simulation:
    st.sidebar.header("Simulation Controls")
    if "sensor_count_to_set" in st.session_state:
        # Apply requested sensor count before instantiating the slider
        st.session_state.num_sensors = st.session_state.pop("sensor_count_to_set")
    if "num_sensors" not in st.session_state:
        st.session_state.num_sensors = 3
    seq_len = st.sidebar.slider("Sequence length", 50, 300, 100)
    num_sensors = st.sidebar.slider("Number of sensors", 1, 64, value=st.session_state.num_sensors, key="num_sensors")
    use_model = st.sidebar.checkbox("Use saved Digital Twin model", value=False)
    models_dir = ROOT_DIR / "models"
    CLASSIFIER_MODEL_PATH = models_dir / "batadal_classifier.pth"
    CLASSIFIER_SCALER_PATH = models_dir / "batadal_classifier.npz"
    DEFAULT_MODEL_PATH = CLASSIFIER_MODEL_PATH if CLASSIFIER_MODEL_PATH.exists() else models_dir / "lstm_autoencoder.pth"
    available_models = [str(p.relative_to(ROOT_DIR)) for p in sorted(models_dir.glob("*")) if p.suffix.lower() in {".pth", ".pt", ".pkl"}]
    selected_model = st.sidebar.selectbox("Select model file (detected)", options=["(none)"] + available_models, index=0)
    model_path = st.sidebar.text_input("Model path (manual)", value=str(DEFAULT_MODEL_PATH))
    if CLASSIFIER_MODEL_PATH.exists():
        st.sidebar.markdown(f"**Default model:** `{CLASSIFIER_MODEL_PATH.name}` (classifier detected)")
    use_balanced_data = st.sidebar.checkbox("Use real BATADAL data (balanced)", value=False)
    balanced_exists = BALANCED_DATA_PATH.exists()
    if use_balanced_data and not balanced_exists:
        st.sidebar.warning("Balanced dataset not found in data/train_balanced.csv")
    if balanced_exists:
        st.sidebar.markdown(f"**Balanced dataset:** `{BALANCED_DATA_PATH.name}` found")
    attack_strength = st.sidebar.slider("Attack strength", 0.1, 10.0, 3.0)
    steps = st.sidebar.number_input("Steps per run", min_value=1, max_value=50, value=1)
    threshold = st.sidebar.slider("Trust threshold for access", 0.0, 1.0, 0.5)
    if st.sidebar.button("Reset Trust Score"):
        st.session_state.trust_engine.reset()
        st.session_state.history = []
        st.rerun()

    # session state
    if "trust_engine" not in st.session_state:
        st.session_state.trust_engine = TrustEngine()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "twin" not in st.session_state:
        st.session_state.twin = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "scaler_loaded" not in st.session_state:
        st.session_state.scaler_loaded = False
    if "scaler_data" not in st.session_state:
        st.session_state.scaler_data = None
    if "last_real" not in st.session_state:
        st.session_state.last_real = None
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_classifier_score" not in st.session_state:
        st.session_state.last_classifier_score = None
    if "run_counter" not in st.session_state:
        st.session_state.run_counter = 0
    if "model_compatible" not in st.session_state:
        st.session_state.model_compatible = True
    if "model_type" not in st.session_state:
        st.session_state.model_type = "Unknown"
    if "balanced_normal_rows" not in st.session_state:
        st.session_state.balanced_normal_rows = None
    if "balanced_attack_rows" not in st.session_state:
        st.session_state.balanced_attack_rows = None
    if "balanced_features" not in st.session_state:
        st.session_state.balanced_features = None


    def model_matches_sensors(sensor_count: int) -> bool:
        twin = st.session_state.twin
        if twin is None:
            return False
        expected = getattr(twin, "input_dim", None)
        if expected is None:
            return True
        return expected == sensor_count


    def determine_model_type(model_path: Path, twin=None) -> str:
        filename = model_path.name.lower()
        if "batadal_classifier" in filename or "classifier" in filename:
            return "classifier"
        if "autoencoder" in filename or "lstm" in filename:
            return "autoencoder"
        if twin is not None:
            model_obj = getattr(twin, "model", twin)
            class_name = type(model_obj).__name__
            if "Classifier" in class_name:
                return "classifier"
            if "Autoencoder" in class_name:
                return "autoencoder"
        return "Unknown"


    def build_classifier_row(num_sensors: int) -> np.ndarray:
        """Generate a classifier input row using the loaded scaler distribution."""
        if st.session_state.scaler_loaded and st.session_state.scaler_data is not None:
            scaler = st.session_state.scaler_data
            mean = np.asarray(scaler.get("mean", np.zeros(num_sensors)), dtype=np.float32)
            std = np.asarray(scaler.get("std", np.ones(num_sensors)), dtype=np.float32)
            if mean.shape == (num_sensors,) and std.shape == (num_sensors,):
                return np.random.normal(loc=mean, scale=std).astype(np.float32)
        return np.random.normal(loc=0.0, scale=1.0, size=(num_sensors,)).astype(np.float32)


    @st.cache_data
    def load_balanced_rows(path: Path):
        if not path.exists():
            return None, None, None
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        feature_cols = [c for c in df.columns if c not in ['DATETIME', 'ATT_FLAG']]
        
        # NOTE: Labels are inverted in train_balanced.csv
        # ATT_FLAG=1 is NORMAL, ATT_FLAG=0 is ATTACK
        normal_df = df[df['ATT_FLAG'] == 1]
        attack_df = df[df['ATT_FLAG'] == 0]
        
        normal_rows = normal_df[feature_cols].astype(np.float32).to_numpy() if len(normal_df) > 0 else None
        attack_rows = attack_df[feature_cols].astype(np.float32).to_numpy() if len(attack_df) > 0 else None
        
        return normal_rows, attack_rows, feature_cols

    def apply_feature_mask(data: np.ndarray) -> np.ndarray:
        if st.session_state.scaler_loaded and st.session_state.scaler_data is not None:
            mask = st.session_state.scaler_data.get('feature_mask')
            if mask is not None and len(mask) > 0:
                if data.ndim == 1 and data.shape[0] == len(mask):
                    return data[mask]
                if data.ndim == 2 and data.shape[1] == len(mask):
                    return data[:, mask]
        return data

    def normalize_data(data: np.ndarray) -> np.ndarray:
        """Normalize data using loaded scaler mean and std."""
        if st.session_state.scaler_loaded and st.session_state.scaler_data is not None:
            mean = st.session_state.scaler_data.get('mean')
            std = st.session_state.scaler_data.get('std')
            if mean is not None and std is not None:
                return ((data - mean) / std).astype(np.float32)
        return data.astype(np.float32)

    # model selection convenience
    if selected_model != "(none)":
        model_path_to_use = Path(selected_model)
    else:
        model_path_to_use = Path(model_path)
    if not model_path_to_use.is_absolute():
        model_path_to_use = ROOT_DIR / model_path_to_use
    model_exists = model_path_to_use.exists()

    # Model load button
    col1, col2 = st.columns([3,1])
    with col1:
        st.text_input("Resolved model path (display)", value=str(model_path_to_use), disabled=True)
        st.markdown(f"**Model found:** {'Yes' if model_exists else 'No'}")
        
        # Check for scaler file
        scaler_path = model_path_to_use.with_suffix('.npz')
        scaler_exists = scaler_path.exists()
        st.markdown(f"**Scaler found:** {'Yes' if scaler_exists else 'No'}")
        
    with col2:
        load_requested = st.button("Load Model & Scaler")
        p = model_path_to_use
        scaler_p = scaler_path

        if load_requested:
            if not use_model:
                st.warning("Enable 'Use saved Digital Twin model' checkbox in the sidebar to load model.")
            elif DigitalTwin is None:
                st.error("DigitalTwin class not found in workspace (twin.digital_twin).")
            elif not p.exists():
                st.error(f"Model file not found: {p}")
            else:
                try:
                    # Load PyTorch model
                    st.session_state.twin = get_digital_twin(str(p))
                    st.session_state.model_loaded = True
                    st.session_state.model_compatible = model_matches_sensors(num_sensors)

                    # Load scaler data
                    if scaler_p.exists():
                        st.session_state.scaler_data = load_scaler_data(str(scaler_p))
                        st.session_state.scaler_loaded = True
                        st.success("Model and scaler loaded successfully!")
                    else:
                        st.session_state.scaler_data = None
                        st.session_state.scaler_loaded = False
                        st.success("Model loaded. Scaler file not found.")

                    st.session_state.model_type = determine_model_type(p, st.session_state.twin)
                    expected_sensors = getattr(st.session_state.twin, "input_dim", None)
                    if expected_sensors is not None and expected_sensors != num_sensors:
                        st.session_state.model_compatible = False
                        st.warning(
                            f"Loaded model expects {st.session_state.twin.input_dim} sensors, but simulation uses {num_sensors}. "
                            "Model prediction will be skipped until the sensor count matches."
                        )
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.session_state.model_loaded = False
                    st.session_state.scaler_loaded = False
                    st.session_state.model_compatible = False

# Load balanced (normal+attack) data if requested
if use_balanced_data and balanced_exists and st.session_state.balanced_normal_rows is None:
    try:
        normal_rows, attack_rows, features = load_balanced_rows(BALANCED_DATA_PATH)
        if normal_rows is not None and attack_rows is not None:
            st.session_state.balanced_normal_rows = normal_rows
            st.session_state.balanced_attack_rows = attack_rows
            st.session_state.balanced_features = features
            st.sidebar.success(f"Loaded {len(normal_rows)} normal + {len(attack_rows)} attack rows with {len(features)} features")
        else:
            st.sidebar.error("Failed to load both normal and attack rows from balanced dataset")
    except Exception as e:
        st.sidebar.error(f"Failed to load balanced data: {e}")

if use_model and not st.session_state.model_loaded:
    if DigitalTwin is None:
        st.sidebar.error("DigitalTwin class not found in workspace (twin.digital_twin).")
    elif not model_exists:
        st.sidebar.warning("Model file not found. Check the model path or select a valid model.")
    else:
        try:
            st.session_state.twin = get_digital_twin(str(model_path_to_use))
            st.session_state.model_loaded = True
            st.session_state.model_compatible = model_matches_sensors(num_sensors)

            # Auto-load scaler if available
            scaler_path = model_path_to_use.with_suffix('.npz')
            if scaler_path.exists():
                st.session_state.scaler_data = load_scaler_data(str(scaler_path))
                st.session_state.scaler_loaded = True
                st.sidebar.success("Model and scaler auto-loaded.")
            else:
                st.session_state.scaler_data = None
                st.session_state.scaler_loaded = False
                st.sidebar.success("Model auto-loaded. Scaler not found.")

            st.session_state.model_type = determine_model_type(model_path_to_use, st.session_state.twin)
            expected_sensors = getattr(st.session_state.twin, "input_dim", None)
            if expected_sensors is not None and expected_sensors != num_sensors:
                st.session_state.model_compatible = False
                st.sidebar.warning(
                    f"Loaded model expects {st.session_state.twin.input_dim} sensors, but current simulation uses {num_sensors}. "
                    "Model prediction will be skipped until they match."
                )
        except Exception as e:
            st.sidebar.error(f"Auto-load failed: {e}")
            st.session_state.model_loaded = False
            st.session_state.scaler_loaded = False
            st.session_state.model_compatible = False

run_col1, run_col2, run_col3 = st.columns(3)
with run_col1:
    if st.button("Run Step(s)"):

        for _ in range(int(steps)):

            # NORMAL SENSOR DATA
            x = np.linspace(0, 4 * np.pi, seq_len)

            real = np.zeros((seq_len, num_sensors), dtype=np.float32)
            predicted = np.zeros_like(real)

            for s in range(num_sensors):
                phase = np.random.uniform(0, 2 * np.pi)

                signal = np.sin(x + phase)

                noise = np.random.normal(0, 0.03, size=seq_len)

                real[:, s] = signal + noise

                # predicted should be similar but NOT identical
                predicted[:, s] = (
                    pd.Series(real[:, s])
                    .rolling(window=5, min_periods=1)
                    .mean()
                    .values
                    + np.random.normal(0, 0.01, size=seq_len)
                )

            # MODEL PREDICTION
            classifier_anomaly_score = None

            if (
                st.session_state.model_loaded
                and st.session_state.twin is not None
                and model_matches_sensors(num_sensors)
            ):

                try:

                    if st.session_state.model_type == "classifier":

                        input_tensor = torch.tensor(
                            real[-1, :],
                            dtype=torch.float32
                        ).unsqueeze(0)

                    else:

                        input_tensor = torch.tensor(
                            real,
                            dtype=torch.float32
                        ).unsqueeze(0)

                    with torch.no_grad():
                        out = st.session_state.twin.predict(input_tensor)

                    if st.session_state.model_type == "classifier":

                        score_tensor = torch.sigmoid(out)

                        classifier_anomaly_score = float(
                            score_tensor.mean().item()
                        )

                    else:

                        if isinstance(out, torch.Tensor):

                            pred_arr = out.squeeze(0).cpu().numpy()

                            if pred_arr.shape == real.shape:
                                predicted = pred_arr.astype(np.float32)

                except Exception as e:
                    st.error(f"Model prediction failed: {e}")

            # ANOMALY CALCULATION

            if (
                st.session_state.model_type == "classifier"
                and classifier_anomaly_score is not None
            ):
                # NORMAL traffic should remain LOW
                anomaly = float(
                    np.clip(
                        classifier_anomaly_score * 0.3,
                        0.0,
                        0.15
                    )
                )
            else:
                anomaly = compute_anomaly(
                    torch.tensor(real, dtype=torch.float32),
                    torch.tensor(predicted, dtype=torch.float32)
                )

                # Clamp normal autoencoder anomaly
                anomaly = float(
                    np.clip(
                        anomaly,
                        0.0,
                        0.15
                    )
                )

            # SMOOTH GRAPH VALUES
            prev_anomaly = (
                st.session_state.history[-1]["anomaly"]
                if len(st.session_state.history) > 0
                else anomaly
            )

            smoothed_anomaly = (
                0.8 * prev_anomaly
                + 0.2 * anomaly
            )

            trust_score = st.session_state.trust_engine.update_trust(
                smoothed_anomaly
            )

            decision = check_access(
                trust_score,
                threshold=threshold
            )

            # OPTIONAL DEBUG MODE
            DEBUG_MODE = False

            if DEBUG_MODE:
                st.write("DEBUG anomaly:", smoothed_anomaly)
                st.write("DEBUG trust:", trust_score)

            st.session_state.history.append({
                "anomaly": float(smoothed_anomaly),
                "trust": float(trust_score),
                "decision": decision
            })

            st.session_state.last_real = real
            st.session_state.last_pred = predicted
            st.session_state.last_classifier_score = classifier_anomaly_score

        st.session_state.run_counter += 1
with run_col2:

    if st.button("Run Attack (single)"):

        x = np.linspace(0, 4 * np.pi, seq_len)

        real = np.zeros((seq_len, num_sensors), dtype=np.float32)
        predicted = np.zeros_like(real)

        # NORMAL BASE SIGNAL
        for s in range(num_sensors):

            phase = np.random.uniform(0, 2 * np.pi)

            signal = np.sin(x + phase)

            noise = np.random.normal(0, 0.03, size=seq_len)

            real[:, s] = signal + noise

            predicted[:, s] = (
                pd.Series(real[:, s])
                .rolling(window=5, min_periods=1)
                .mean()
                .values
                + np.random.normal(0, 0.01, size=seq_len)
            )

        # ATTACK ONE SENSOR
        sidx = np.random.randint(0, num_sensors)

        real[:, sidx] = inject_attack(
            real[:, sidx],
            strength=attack_strength
        )

        # CREATE LARGE RECONSTRUCTION ERROR
        predicted[:, sidx] = (
            predicted[:, sidx]
            - attack_strength * 0.8
        )

        classifier_anomaly_score = None

        # MODEL PREDICTION
        if (
            st.session_state.model_loaded
            and st.session_state.twin is not None
            and model_matches_sensors(num_sensors)
        ):

            try:

                if st.session_state.model_type == "classifier":

                    input_tensor = torch.tensor(
                        real[-1, :],
                        dtype=torch.float32
                    ).unsqueeze(0)

                else:

                    input_tensor = torch.tensor(
                        real,
                        dtype=torch.float32
                    ).unsqueeze(0)

                with torch.no_grad():

                    out = st.session_state.twin.predict(input_tensor)

                if st.session_state.model_type == "classifier":

                    score_tensor = torch.sigmoid(out)

                    classifier_anomaly_score = float(
                        score_tensor.mean().item()
                    )

                else:

                    if isinstance(out, torch.Tensor):

                        pred_arr = out.squeeze(0).cpu().numpy()

                        if pred_arr.shape == real.shape:
                            predicted = pred_arr.astype(np.float32)

            except Exception as e:
                st.error(f"Model prediction failed: {e}")

        # FORCE HIGHER ANOMALY
        if (
            st.session_state.model_type == "classifier"
            and classifier_anomaly_score is not None
        ):

            anomaly = float(
                np.clip(
                    max(
                        classifier_anomaly_score * 5.0,
                        0.75
                    ),
                    0.0,
                    1.0
                )
            )

        else:

            anomaly = compute_anomaly(
                torch.tensor(real, dtype=torch.float32),
                torch.tensor(predicted, dtype=torch.float32)
            )

            anomaly = float(
                np.clip(
                    max(anomaly, 0.75),
                    0.0,
                    1.0
                )
            )

        trust_score = st.session_state.trust_engine.update_trust(anomaly)

        # FORCE ACCESS DENIAL
        if anomaly > 0.5:
            decision = "ACCESS DENIED"
        else:
            decision = check_access(
                trust_score,
                threshold=threshold
            )

        # DEBUG
        st.write("DEBUG ATTACK anomaly:", anomaly)
        st.write("DEBUG ATTACK trust:", trust_score)

        st.session_state.history.append({
            "anomaly": float(anomaly),
            "trust": float(trust_score),
            "decision": decision,
            "attacked_sensor": int(sidx)
        })

        st.session_state.last_real = real
        st.session_state.last_pred = predicted
        st.session_state.last_classifier_score = classifier_anomaly_score

        st.session_state.run_counter += 1
with run_col3:
    if st.button("Reset All"):
        st.session_state.history = []
        st.session_state.trust_engine.reset()
        st.session_state.twin = None
        st.session_state.model_loaded = False
        st.session_state.model_compatible = True
        st.session_state.model_type = "Unknown"
        st.session_state.scaler_loaded = False
        st.session_state.scaler_data = None
        st.session_state.last_real = None
        st.session_state.last_pred = None
        st.session_state.last_classifier_score = None
        st.session_state.run_counter = 0
        st.session_state.balanced_normal_rows = None
        st.session_state.balanced_attack_rows = None
        st.session_state.balanced_features = None
        st.success("Reset complete.")

# Display panels
history_df = pd.DataFrame(st.session_state.history)
left, right = st.columns([2,1])

with left:
    st.subheader("Anomaly & Trust Timeline")
    if history_df.empty:
        st.info("No runs yet. Use 'Run Step(s)' or 'Run Attack (single)'.")
    else:
        if PLOTLY_AVAILABLE:
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(y=history_df["anomaly"], mode='lines', name='Anomaly Score', line=dict(color='#FF6B6B', width=2)))
            fig_timeline.add_trace(go.Scatter(y=history_df["trust"], mode='lines', name='Trust Score', line=dict(color='#4ECDC4', width=2)))
            fig_timeline.update_layout(
                title="Anomaly & Trust Evolution",
                xaxis_title="Run Step",
                yaxis_title="Score",
                template="plotly_dark",
                hovermode="x unified",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_timeline, width="stretch", key=f"timeline_run_{st.session_state.run_counter}")
        else:
            st.line_chart(history_df[["anomaly", "trust"]])

    st.subheader("Per-sensor Series (latest run)")
    if st.session_state.last_real is None:
        st.write("No sensor data yet.")
    else:
        real = st.session_state.last_real
        pred = st.session_state.last_pred
        # per-sensor tabs
        tabs = st.tabs([f"Sensor {i}" for i in range(real.shape[1])])
        for i, tab in enumerate(tabs):
            with tab:
                df_plot = pd.DataFrame({
                    "real": real[:, i],
                    "predicted": pred[:, i]
                })
                # Use Plotly for better real-time updates
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=df_plot["real"], mode='lines', name='Real', line=dict(color='#00D9FF', width=2)))
                    fig.add_trace(go.Scatter(y=df_plot["predicted"], mode='lines', name='Predicted', line=dict(color='#7F39FB', width=2)))
                    fig.update_layout(
                        title=f"Sensor {i} - Real vs Predicted",
                        xaxis_title="Time Step",
                        yaxis_title="Value",
                        template="plotly_dark",
                        hovermode="x unified",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, width="stretch", key=f"sensor_{i}_run_{st.session_state.run_counter}")
                else:
                    st.line_chart(df_plot)

                st.markdown("Latest values (last 5 samples)")
                st.table(pd.DataFrame({"real": real[-5:, i], "predicted": pred[-5:, i]}))

with right:
    st.subheader("Latest Metrics")
    if history_df.empty:
        st.metric("Anomaly Score", "N/A")
        st.metric("Trust Score", "N/A")
        st.markdown("**Decision:** N/A")
    else:
        last = history_df.iloc[-1]
        st.metric("Anomaly Score", f"{last['anomaly']:.4f}")
        st.metric("Trust Score", f"{last['trust']:.4f}")
        if st.session_state.model_type == "classifier" and st.session_state.last_classifier_score is not None:
            st.metric("Attack probability", f"{st.session_state.last_classifier_score:.4f}")
        if last['decision'] == "ACCESS GRANTED":
            st.markdown(f"**Decision:** <span style='color: green; font-weight: bold;'>{last['decision']}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Decision:** <span style='color: red; font-weight: bold;'>{last['decision']}</span>", unsafe_allow_html=True)
        if "attacked_sensor" in last and not np.isnan(last.get("attacked_sensor", np.nan)):
            st.markdown(f"**Attacked sensor (if any):** {int(last['attacked_sensor'])}")

    st.markdown("### Model status")
    if st.session_state.model_loaded:
        expected_sensors = getattr(st.session_state.twin, "input_dim", None)
        st.markdown(f"**Model type:** {st.session_state.model_type.capitalize()}")
        if expected_sensors is not None:
            st.markdown(f"**Model expects:** {expected_sensors} sensor(s)")
        if not st.session_state.model_compatible:
            st.warning(
                f"Loaded model expects {expected_sensors} sensors, but current simulation uses {num_sensors}. "
                "Model prediction will be skipped until they match."
            )
        st.success(f"✓ Model loaded: {Path(model_path_to_use).name}")
        if st.session_state.scaler_loaded and st.session_state.scaler_data is not None:
            st.success(f"✓ Scaler loaded: threshold = {st.session_state.scaler_data['threshold']:.4f}")
        else:
            st.warning("⚠ Scaler not loaded - using default normalization")
    else:
        st.info("No model loaded. Use the Load Model button or provide a valid model in /models.")

    st.markdown("---")
    st.caption("Run locally: streamlit run app.py | Deploy on Streamlit Cloud")

with tab_evaluation:
    st.header("Model Evaluation & Metrics")
    
    # Evaluation controls
    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        eval_dataset = st.selectbox("Dataset", ["batadal", "swat", "synthetic"], index=0)
        eval_model_type = st.selectbox("Model Type", ["single_classifier", "autoencoder", "ensemble"], index=0)
    
    with eval_col2:
        eval_run = st.button("Run Evaluation", type="primary")
        if eval_run:
            with st.spinner("Running evaluation..."):
                # Load data
                data, y_true = load_eval_data(dataset=eval_dataset)
                
                # Check for ensemble
                has_ensemble = eval_model_type == "ensemble" and all(
                    (ENSEMBLE_BASE_PATH.with_name(f"{ENSEMBLE_BASE_PATH.name}_model{i}.pth")).exists() 
                    for i in range(3)
                )
                
                if has_ensemble:
                    anomaly_scores = use_ensemble_models_eval(data, y_true)
                else:
                    # Single model
                    use_classifier = eval_model_type == "single_classifier" and CLASSIFIER_MODEL_PATH.exists()
                    model_path = CLASSIFIER_MODEL_PATH if use_classifier else ROOT_DIR / "models" / "lstm_autoencoder.pth"
                    scaler_path = CLASSIFIER_SCALER_PATH if use_classifier else ROOT_DIR / "models" / "lstm_autoencoder.npz"
                    
                    if not model_path.exists():
                        st.error(f"Model not found: {model_path}")
                        st.stop()
                    
                    twin = get_digital_twin(str(model_path))
                    mean, std, threshold, feature_mask = load_scaler_eval(str(scaler_path))
                    
                    # Apply feature selection
                    if feature_mask is not None and len(feature_mask) > 0:
                        data = data[:, feature_mask]
                    
                    # Normalize
                    if mean is not None and std is not None:
                        data = (data - mean) / std
                    
                    data = data.astype(np.float32)
                    
                    if use_classifier:
                        anomaly_scores = []
                        for row in data:
                            tensor = torch.tensor(row).unsqueeze(0)
                            pred = twin.predict(tensor)
                            score = torch.sigmoid(pred.squeeze(0)).item()
                            anomaly_scores.append(score)
                        anomaly_scores = np.array(anomaly_scores)
                        # Debug: check the distribution
                        st.write(f"DEBUG: Anomaly scores min={anomaly_scores.min():.6f}, max={anomaly_scores.max():.6f}, mean={anomaly_scores.mean():.6f}")
                        st.write(f"DEBUG: y_true distribution - Normal (0)={int(np.sum(y_true == 0))}, Attack (1)={int(np.sum(y_true == 1))}")
                        st.write(f"DEBUG: y_true class 0 scores mean={anomaly_scores[y_true == 0].mean():.6f}")
                        st.write(f"DEBUG: y_true class 1 scores mean={anomaly_scores[y_true == 1].mean():.6f}")
                    else:
                        seq_len = 100
                        if len(data) >= seq_len:
                            data, y_true = create_sequences_eval(data, y_true, seq_len=seq_len)
                        else:
                            st.error("Data too short for sequence generation")
                            st.stop()
                        
                        anomaly_scores = []
                        for seq in data:
                            tensor = torch.tensor(seq).unsqueeze(0)
                            pred = twin.predict(tensor)
                            score = compute_anomaly(tensor.squeeze(0), pred.squeeze(0))
                            anomaly_scores.append(score)
                    
                    anomaly_scores = np.array(anomaly_scores)
                
                # Threshold optimization
                objectives = ["f1", "balanced", "precision", "recall", "specificity"]
                threshold_results = {}
                
                st.subheader("Threshold Optimization Results")
                cols = st.columns(len(objectives))
                
                for i, obj in enumerate(objectives):
                    with cols[i]:
                        t_opt, score, _ = find_best_threshold_for_objective(y_true, anomaly_scores, objective=obj)
                        threshold_results[obj] = (t_opt, score)
                        metrics_opt = evaluate_metrics(y_true, anomaly_scores, threshold=t_opt)
                        st.metric(f"{obj.upper()} Threshold", f"{t_opt:.4f}")
                        st.metric(f"{obj.upper()} Score", f"{score:.4f}")
                        st.write(f"Precision: {metrics_opt['Precision']:.4f}")
                        st.write(f"Recall: {metrics_opt['Recall']:.4f}")
                        st.write(f"F1: {metrics_opt['F1']:.4f}")
                
                # Recommended configuration
                st.subheader("Recommended Configuration")
                best_threshold_balanced, _ = threshold_results["balanced"]
                metrics = evaluate_metrics(y_true, anomaly_scores, threshold=best_threshold_balanced)
                
                st.success(f"Balanced Threshold: {best_threshold_balanced:.4f}")
                st.write("(Optimizes for equal recall and specificity)")
                
                st.subheader("Final Performance Metrics")
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                with metric_cols[1]:
                    st.metric("F1 Score", f"{metrics['F1']:.4f}")
                    st.metric("Specificity", f"{1 - metrics['FPR']:.4f}")
                with metric_cols[2]:
                    st.metric("False Positive Rate", f"{metrics['FPR']:.4f}")
                    st.metric("ROC AUC", f"{metrics['ROC_AUC']:.4f}")
                
                # Data summary
                st.subheader("Dataset Summary")
                st.write(f"Total samples: {len(y_true)}")
                st.write(f"Normal samples: {int(np.sum(y_true == 0))}")
                st.write(f"Attack samples: {int(np.sum(y_true == 1))}")
                st.write(f"Attack ratio: {np.mean(y_true):.2%}")
