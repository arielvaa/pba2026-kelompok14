"""
Emotion Classification App - Gradio Interface
Deployed on Hugging Face Spaces
"""
import gradio as gr
import joblib
import re
import pandas as pd
import numpy as np

# Load model and vectorizer
model = joblib.load('svm_(linearsvc)_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function (same as training)
def clean_text(text):
    """Clean text for prediction"""
    if not text or text.strip() == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Prediction function
def predict_emotion(text):
    """Predict emotion from text"""
    if not text or text.strip() == "":
        return "Please enter some text!", None
    
    # Clean text
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return "Text is empty after cleaning. Please enter meaningful text.", None
    
    # Vectorize
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    
    # Get decision function scores for confidence (if available)
    try:
        scores = model.decision_function(text_tfidf)[0]
        
        # Normalize scores to probabilities (softmax-like)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        # Get all classes
        classes = model.classes_
        
        # Create confidence dict
        confidence = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        # Sort by confidence
        confidence_sorted = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))
        
        return prediction, confidence_sorted
    except:
        return prediction, None

# Example texts
examples = [
    ["I am so happy today! This is the best day ever!"],
    ["I am really angry about this situation. This is unacceptable!"],
    ["I feel so sad and lonely right now."],
    ["I am scared and worried about what will happen next."],
    ["This is absolutely disgusting and horrible."],
    ["I am so proud of what we have achieved together!"],
    ["I feel surprised by this unexpected turn of events."],
    ["I am bored and have nothing interesting to do."],
]

# Create Gradio interface
with gr.Blocks(title="Emotion Classification", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 Emotion Classification App
        ### Classify emotions from text using Machine Learning
        
        This app uses a **Support Vector Machine (SVM)** model trained on 20 different emotions.
        Simply enter your text below and see what emotion it conveys!
        
        **Model Details:**
        - Algorithm: Linear SVM (LinearSVC)
        - Features: TF-IDF (5000 features, unigrams + bigrams)
        - Training Data: 79,595 samples across 20 emotions
        - Performance: ~XX% F1-Score on test set
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type something here... (e.g., 'I am so happy today!')",
                lines=5
            )
            
            predict_btn = gr.Button("🔮 Predict Emotion", variant="primary", size="lg")
            clear_btn = gr.ClearButton([text_input], value="Clear", size="sm")
            
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label="Try these examples:"
            )
        
        with gr.Column(scale=1):
            emotion_output = gr.Textbox(
                label="Predicted Emotion",
                interactive=False
            )
            
            confidence_output = gr.Label(
                label="Confidence Scores (Top 10)",
                num_top_classes=10
            )
    
    gr.Markdown(
        """
        ---
        ### 📊 About the Model
        
        This emotion classifier was trained on a dataset of ~80K text samples labeled with 20 different emotions:
        - **Positive:** joy, love, optimism, gratitude, pride, amusement, etc.
        - **Negative:** anger, sadness, fear, disgust, disappointment, grief, etc.
        - **Neutral:** surprise, curiosity, confusion, realization, etc.
        
        The model uses **TF-IDF vectorization** to convert text into numerical features and 
        a **Linear Support Vector Machine** for classification.
        
        ### 🛠️ Technical Stack
        - **Framework:** Gradio
        - **ML Library:** scikit-learn
        - **Model:** LinearSVC
        - **Deployment:** Hugging Face Spaces
        
        ---
        *Made with ❤️ for NLP Assignment*
        """
    )
    
    # Connect button to function
    predict_btn.click(
        fn=predict_emotion,
        inputs=text_input,
        outputs=[emotion_output, confidence_output]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
