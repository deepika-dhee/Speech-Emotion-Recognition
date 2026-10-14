# 🎙️ Speech Emotion Recognition

## 📌 Project Overview
Speech Emotion Recognition (SER) is a machine learning project that identifies human emotions from speech audio. This project extracts meaningful features from audio recordings and predicts the emotion expressed in the speaker's voice.

The project uses the **Toronto Emotional Speech Set (TESS)** dataset and includes a **Streamlit web application** for interactive emotion prediction.

## 🎯 Objectives
- Analyze speech audio and extract relevant features.
- Train a machine learning model to classify emotions.
- Predict emotions from uploaded audio samples.
- Build an interactive web interface using Streamlit.
- Evaluate model performance using standard classification metrics.

## 🛠️ Technologies Used
- Python
- NumPy and Pandas
- Librosa
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook

## 📂 Dataset
The project uses the TESS speech emotion dataset containing audio recordings representing different emotions.

Emotion categories supported by the implementation may include:
- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## ⚙️ Project Workflow
1. Load and explore the speech audio dataset.
2. Extract audio features such as MFCCs, where implemented.
3. Prepare the features for machine learning.
4. Split the dataset into training and testing sets.
5. Train the emotion classification model.
6. Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Use the trained model in the Streamlit application to predict emotions from audio.

## 🌐 Streamlit Web Application
The Streamlit interface allows users to interact with the Speech Emotion Recognition system through a web-based interface.

**Application features may include:**
- Uploading speech audio.
- Processing audio for feature extraction.
- Predicting the emotion using the trained model.
- Displaying the predicted emotion.

Only include the features that are actually implemented in your application.

## 📁 Project Structure

```text
Speech-Emotion-Recognition/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── speech_emotion_recognition.ipynb
└── artifacts/
    └── emotion_model.joblib
```

*This is a suggested structure. Keep only files and folders that exist in your repository. The `artifacts/` folder is needed only if your app loads a separately saved model.*

## 🚀 Installation and Setup

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 📊 Model Evaluation
The model can be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Add the actual results from your experiments here.

## ⚠️ Limitations
- Performance depends on the dataset and audio quality.
- Background noise, accents, and differences between speakers can affect predictions.
- Performance on TESS data may differ from performance on real-world speech.

## 🔮 Future Enhancements
- Deploy the Streamlit application online.
- Evaluate on more diverse speech datasets.
- Improve robustness to noise and speaker variation.
- Explore deep learning approaches for emotion classification.

## 🙏 Acknowledgements
Thanks to the creators of the TESS dataset and the open-source Python community.

