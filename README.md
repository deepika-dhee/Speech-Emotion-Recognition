# Speech Emotion Recognition 🎙️

## Project Overview
Speech Emotion Recognition (SER) is a machine learning project that identifies human emotions from speech audio. It analyzes audio features and predicts the emotion expressed in the speaker's voice.

This project uses the **Toronto Emotional Speech Set (TESS)** dataset to classify speech into different emotional categories.

## Objectives
- Extract meaningful features from speech audio.
- Train a machine learning model to recognize emotions.
- Predict emotions from audio samples.
- Evaluate model performance using classification metrics.

## Technologies Used
- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Dataset
The project uses the TESS speech emotion dataset, which contains audio recordings representing different emotions.

The emotion categories used in this project are:
- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

*Note: The exact emotion categories depend on the dataset files used during training.*

## Project Workflow
1. Load and explore the audio dataset.
2. Extract audio features, such as Mel-Frequency Cepstral Coefficients (MFCCs), if used in the implementation.
3. Prepare the extracted features for training.
4. Split the dataset into training and testing sets.
5. Train a machine learning classification model.
6. Evaluate the model using accuracy, precision, recall, and F1-score.
7. Predict emotions from audio samples.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run
Open the Jupyter Notebook included in this repository and run the cells in order.

```bash
jupyter notebook
```

## Model Evaluation
The model can be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Add your actual evaluation results after running the notebook.

## Limitations
- The model is trained on a specific speech emotion dataset.
- Performance may vary with different speakers, recording devices, background noise, and accents.
- Results on the TESS dataset may not represent performance on real-world speech.

## Future Improvements
- Build a Streamlit web application for audio-based emotion prediction.
- Test the model on additional speech datasets.
- Improve robustness to noise and different speakers.
- Explore deep learning models for speech emotion recognition.

## Author
**Deepika P**

AI & Machine Learning Student

## Acknowledgements
Thanks to the creators of the TESS dataset and the open-source Python machine learning community.
