---
title: Emotion Classification
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 Emotion Classification App

Classify emotions from text using Machine Learning!

## Model Details

- **Algorithm:** Linear SVM (LinearSVC)
- **Features:** TF-IDF vectorization (5000 features, unigrams + bigrams)
- **Training Data:** 79,595 samples across 20 emotions
- **Emotions Detected:** joy, anger, sadness, fear, love, surprise, gratitude, optimism, and more!

## How It Works

1. **Text Input:** Enter any text you want to analyze
2. **Preprocessing:** Text is cleaned (lowercase, remove URLs, special characters)
3. **Vectorization:** Text converted to TF-IDF features
4. **Prediction:** SVM model classifies the emotion
5. **Output:** See the predicted emotion with confidence scores!

## Examples

Try these examples:
- "I am so happy today! This is the best day ever!"
- "I am really angry about this situation."
- "I feel so sad and lonely right now."
- "I am scared and worried about what will happen."

## Technical Stack

- **Framework:** Gradio
- **ML Library:** scikit-learn  
- **Model:** Linear Support Vector Machine
- **Deployment:** Hugging Face Spaces

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

## Dataset

Trained on the [Emotion Dataset (20 Emotions)](https://huggingface.co/datasets/shreyaspulle98/emotion-dataset-20-emotions) from Hugging Face.

---

*Made for NLP Assignment 2026*
