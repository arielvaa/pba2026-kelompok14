# 🚀 Deployment Guide - Hugging Face Spaces

## 📋 Prerequisites
- Hugging Face account
- Git installed
- Model files trained (`svm_(linearsvc)_model.pkl` and `tfidf_vectorizer.pkl`)

## 🎯 Deployment Steps

### Option 1: Upload via Web Interface (Easiest)

1. **Create a new Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `emotion-classification` (or your choice)
   - SDK: Select **Gradio**
   - License: MIT
   - Click "Create Space"

2. **Upload Files**
   - Click "Files and versions" tab
   - Upload these files from `apps/ml_app/`:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `svm_(linearsvc)_model.pkl`
     - `tfidf_vectorizer.pkl`

3. **Wait for Build**
   - HF Spaces will automatically detect changes
   - Building takes ~2-5 minutes
   - Once done, your app will be live!

### Option 2: Git Push (Advanced)

1. **Clone your Space**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/emotion-classification
   cd emotion-classification
   ```

2. **Copy Files**
   ```bash
   cp apps/ml_app/app.py .
   cp apps/ml_app/requirements.txt .
   cp apps/ml_app/README.md .
   cp apps/ml_app/*.pkl .
   ```

3. **Commit and Push**
   ```bash
   git add .
   git commit -m "Initial deployment: Emotion classification app"
   git push
   ```

4. **Access Your App**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/emotion-classification`

## 📁 Required Files Structure

```
your-space/
├── app.py                          # Main Gradio app
├── requirements.txt                # Python dependencies
├── README.md                       # Space description
├── svm_(linearsvc)_model.pkl      # Trained model
└── tfidf_vectorizer.pkl           # TF-IDF vectorizer
```

## ⚙️ Configuration

The `README.md` contains metadata for HF Spaces:
```yaml
---
title: Emotion Classification
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
---
```

## 🧪 Test Locally First

Before deploying, test the app locally:

```bash
cd apps/ml_app
pip install -r requirements.txt
python app.py
```

Open the URL shown in terminal (usually http://127.0.0.1:7860)

## 🔧 Troubleshooting

**Issue: Model file not found**
- Make sure `svm_(linearsvc)_model.pkl` and `tfidf_vectorizer.pkl` are in the same directory as `app.py`

**Issue: Build fails**
- Check `requirements.txt` for correct package versions
- Verify all files are uploaded correctly

**Issue: App crashes on prediction**
- Check model compatibility with scikit-learn version
- Verify vectorizer was saved with same sklearn version

## 📊 Model Performance

Update the README.md with your actual F1-score from `reports/tables/ml_model_comparison.csv`:

```markdown
- **Performance:** ~XX% F1-Score on test set
```

## 🎨 Customization

**Change Theme:**
In `app.py`, modify:
```python
with gr.Blocks(title="Emotion Classification", theme=gr.themes.Soft()) as demo:
```

Options: `gr.themes.Soft()`, `gr.themes.Base()`, `gr.themes.Monochrome()`

**Add More Examples:**
In `app.py`, add to the `examples` list:
```python
examples = [
    ["Your example text here"],
    # ... more examples
]
```

**Enable Public Links:**
In `app.py`, change:
```python
demo.launch(share=True)  # Creates public ngrok link
```

## 🌐 Post-Deployment

1. **Test the live app** - Try different inputs
2. **Share the link** - Send to classmates/professor
3. **Monitor usage** - Check HF Spaces dashboard for analytics
4. **Update if needed** - Upload new files to update the app

## 📝 Example Space URL

After deployment, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/emotion-classification
```

Share this link for your assignment submission!

---

Good luck with your deployment! 🚀
