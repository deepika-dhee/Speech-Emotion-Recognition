from pathlib import Path
import json
import tempfile

import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import ConfusionMatrixDisplay

# ---------- CONFIG ----------
st.set_page_config(
    page_title="VoxSense | Emotion AI",
    page_icon="🎙️",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
DATASET = BASE / "Tess" / "features_tess.csv"

FEATURES = [
    name
    for i in range(1, 14)
    for name in (f"mfcc_{i}_mean", f"mfcc_{i}_std")
] + [
    "energy_mean", "energy_std", "zcr_mean",
    "centroid_mean", "bandwidth_mean", "rolloff_mean"
]

# ---------- STYLING ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff);
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #101936, #34276f, #6855d9);
    padding: 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #e0e7ff;
    font-size: 16px;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e0e7ff;
    box-shadow: 0 5px 18px #202b5510;
    margin-bottom: 16px;
}
.result {
    background: linear-gradient(120deg, #e9e7ff, #e0f2fe);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #c7d2fe;
}
.result h2 {
    color: #4338ca;
}
.stButton button {
    background: linear-gradient(90deg, #5145cd, #7c5ce7);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 600;
    min-height: 44px;
}
h2, h3 {
    color: #202b50;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    required = [
        "gda_model.joblib",
        "scaler.joblib",
        "evaluation.json",
        "confusion_matrix.csv",
        "classification_report.csv"
    ]

    missing = [f for f in required if not (ARTIFACTS / f).exists()]
    if missing:
        return None

    with open(ARTIFACTS / "evaluation.json", encoding="utf-8") as f:
        evaluation = json.load(f)

    return {
        "model": joblib.load(ARTIFACTS / "gda_model.joblib"),
        "scaler": joblib.load(ARTIFACTS / "scaler.joblib"),
        "evaluation": evaluation,
        "cm": pd.read_csv(
            ARTIFACTS / "confusion_matrix.csv", index_col=0
        ),
        "report": pd.read_csv(
            ARTIFACTS / "classification_report.csv", index_col=0
        )
    }

def extract_features(path):
    y, sr = librosa.load(path, sr=22050, mono=True)

    if len(y) == 0:
        raise ValueError("The audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
    )

    values = {}
    for i in range(13):
        values[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        values[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    values["energy_mean"] = np.mean(librosa.feature.rms(y=y)[0])
    values["energy_std"] = np.std(librosa.feature.rms(y=y)[0])
    values["zcr_mean"] = np.mean(
        librosa.feature.zero_crossing_rate(y)[0]
    )
    values["centroid_mean"] = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    )
    values["bandwidth_mean"] = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    )
    values["rolloff_mean"] = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    )

    return values

def predict_probabilities(X, model):
    scores = []

    for label in model["classes"]:
        diff = X - model["means"][label]
        distance = np.sum(
            (diff @ model["covariance_inverse"]) * diff,
            axis=1
        )
        scores.append(
            np.log(model["priors"][label])
            - 0.5 * distance
            - 0.5 * model["log_det"]
        )

    return softmax(np.column_stack(scores), axis=1)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🎙️ VoxSense</h1>
    <h3 style="color:#c4b5fd;">Speech Emotion Intelligence</h3>
    <p>Discover the emotion behind a voice using
    Gaussian Discriminant Analysis.</p>
    <p>✦ TESS Dataset &nbsp; ✦ GDA Model &nbsp; ✦ 32 Acoustic Features</p>
</div>
""", unsafe_allow_html=True)

model_data = load_model()

if model_data is None:
    st.error(
        "Trained model files are missing. Place train_model.py in this "
        "folder and run `python train_model.py` first."
    )
    st.stop()

model = model_data["model"]
scaler = model_data["scaler"]

# ---------- NAVIGATION ----------
home, predict, performance, dataset_tab = st.tabs([
    "🏠 Overview",
    "🎧 Predict Emotion",
    "📊 Performance",
    "🗂️ Dataset"
])

with home:
    st.markdown("## Understand the voice")

    a, b, c = st.columns(3)

    for col, emoji, title, desc in [
        (a, "🎧", "Upload", "Choose a clear speech recording."),
        (b, "🧠", "Analyze", "Extract and standardize audio features."),
        (c, "✨", "Discover", "View the predicted emotion and scores.")
    ]:
        with col:
            st.markdown(f"""
            <div class="card">
                <h2>{emoji}</h2>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Emotions supported")

    emotions = {
        "😊": "Happy", "😢": "Sad", "😠": "Angry",
        "😨": "Fear", "🤢": "Disgust", "😲": "Surprise"
    }

    cols = st.columns(6)
    for col, (emoji, name) in zip(cols, emotions.items()):
        col.markdown(
            f"<div class='card' style='text-align:center;'>"
            f"<h2>{emoji}</h2><b>{name}</b></div>",
            unsafe_allow_html=True
        )

    st.info(
        "Predictions are estimates from the trained model, "
        "not definitive measurements of a person's feelings."
    )

with predict:
    st.markdown("## 🎧 Analyze a voice")
    st.write("Upload a short, clear WAV recording.")

    uploaded = st.file_uploader(
        "Choose an audio file", type=["wav"]
    )

    if uploaded:
        st.audio(uploaded)

        if st.button("✨ Predict emotion", use_container_width=True):
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    temp_path = Path(tmp.name)

                try:
                    with st.spinner("Analyzing voice patterns..."):
                        features = extract_features(temp_path)
                        row = pd.DataFrame(
                            [[features[f] for f in FEATURES]],
                            columns=FEATURES
                        )

                        if list(model["feature_names"]) != FEATURES:
                            raise ValueError(
                                "Feature order differs from the trained model. "
                                "Retrain using the matching feature definition."
                            )

                        X = scaler.transform(row.to_numpy(dtype=float))
                        probabilities = predict_probabilities(X, model)[0]
                        classes = list(model["classes"])
                        best = int(np.argmax(probabilities))
                        emotion = str(classes[best])
                        confidence = float(probabilities[best])

                    st.markdown(f"""
                    <div class="result">
                        <h2>✨ Predicted Emotion</h2>
                        <h1 style="color:#4338ca;">{emotion}</h1>
                        <p>Highest estimated class probability:
                        <b>{confidence:.1%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    result = pd.DataFrame({
                        "Emotion": classes,
                        "Probability": probabilities
                    }).sort_values("Probability", ascending=False)

                    st.markdown("### Emotion probability scores")
                    st.bar_chart(
                        result.set_index("Emotion")["Probability"]
                    )
                    st.dataframe(
                        result.style.format({"Probability": "{:.2%}"}),
                        use_container_width=True,
                        hide_index=True
                    )

                finally:
                    temp_path.unlink(missing_ok=True)

            except Exception as e:
                st.error(f"Could not analyze the audio: {e}")

with performance:
    st.markdown("## 📊 Model performance")

    gda = model_data["evaluation"].get("GDA", {})

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f'{gda.get("accuracy", 0):.2%}')
    b.metric("Macro Precision", f'{gda.get("macro_precision", 0):.2%}')
    c.metric("Macro Recall", f'{gda.get("macro_recall", 0):.2%}')
    d.metric("Macro F1-score", f'{gda.get("macro_f1", 0):.2%}')

    st.markdown("### Confusion matrix")

    cm = model_data["cm"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm.to_numpy(),
        display_labels=cm.index.astype(str)
    ).plot(ax=ax, colorbar=False, xticks_rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("View classification report"):
        st.dataframe(model_data["report"], use_container_width=True)

    with st.expander("View all saved model metrics"):
        st.dataframe(
            pd.DataFrame(model_data["evaluation"]).T,
            use_container_width=True
        )

with dataset_tab:
    st.markdown("## 🗂️ Dataset insights")

    candidates = [
        BASE / "data" / "features_tess.csv",
        DATASET,
        Path(r"E:\gda\Tess\features_tess.csv")
    ]
    csv_path = next((p for p in candidates if p.exists()), None)

    if csv_path is None:
        st.warning("Dataset CSV not found. Check the dataset path.")
    else:
        df = pd.read_csv(csv_path)

        if "emotion" not in df.columns:
            st.error("The CSV must contain an 'emotion' column.")
        else:
            a, b, c = st.columns(3)
            a.metric("Audio samples", f"{len(df):,}")
            b.metric("Emotion classes", df["emotion"].nunique())
            b.metric("Acoustic features", len(FEATURES))
            c.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

            st.markdown("### Emotion distribution")
            st.bar_chart(df["emotion"].value_counts())

            with st.expander("Preview dataset"):
                st.dataframe(df.head(10), use_container_width=True)

st.divider()
st.caption("VoxSense · Speech Emotion Recognition · Gaussian Discriminant Analysis")
