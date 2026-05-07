# 🧠 CNN Sentiment Analysis Web App

Bonus assignment: original lab CNN vs improved CNN, with a Streamlit interface.

---

## 📁 File Structure

```
sentiment_app/
├── app.py           ← Streamlit web app  (main entry point)
├── models.py        ← CNN architectures + training + inference
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

You also need your **`glove_6B_100d.txt`** file in the same folder (or set its path in the sidebar).

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place GloVe file
Copy `glove_6B_100d.txt` into the `sentiment_app/` folder.

### 3. Run the app
```bash
cd sentiment_app
streamlit run app.py
```

The browser will open at **http://localhost:8501**

---

## 🏗️ What's Inside

### Tab 1 — Train & Compare
- Click **"🚀 Train Both Models"** in the sidebar
- Both CNNs train on NLTK Twitter Samples (10,000 tweets)
- Side-by-side accuracy / loss metrics
- Live training curves (Plotly)
- Architecture diff table

### Tab 2 — Predict
- Type any sentence → get Positive/Negative prediction + confidence bar
- Switch between Original, Improved, or Both

### Tab 3 — Batch CSV
- Upload a CSV with a `text` column
- Get predictions for all rows
- Download results CSV
- Pie chart of sentiment distribution

### Tab 4 — Dataset Explorer
- Upload a **labelled** dataset (text, label columns)
- See accuracy, confusion matrix, and classification report on **new data**
- Explains why performance might change across domains

---

## 🔧 Model Architectures

### Original CNN (Lab)
```
Embedding(GloVe-100d, frozen)
→ Conv1D(128, kernel=5, relu)
→ GlobalMaxPooling1D
→ Dense(64, relu)
→ Dense(1, sigmoid)
```

### Improved CNN
```
Embedding(GloVe-100d, trainable)   ← fine-tune embeddings
→ Conv1D(128, k=3, relu) ─┐
→ Conv1D(128, k=5, relu) ─┴→ Concatenate   ← dual branches
→ Dropout(0.4)             ← regularisation
→ BatchNormalization        ← stable training
→ Dense(128, relu)
→ Dense(1, sigmoid)
```

**Improvements explained:**

| Change | Benefit |
|--------|---------|
| Trainable GloVe | Adapts vectors to sentiment domain |
| Dual Conv1D (k=3 & k=5) | Captures short AND medium n-gram patterns |
| Dropout(0.4) | Reduces overfitting |
| BatchNormalization | Faster convergence, more stable gradients |
| Dense(128) vs Dense(64) | More capacity for feature fusion |

---

## 📊 Testing on Another Dataset (IMDB)

The IMDB dataset (`imdb_labelled.tsv`) uses these columns: `text \t label`  
Upload it in **Tab 4 → Dataset Explorer**.

Expected behaviour: accuracy may drop vs Twitter because:
- Movie reviews are longer (truncation at MAX_LEN=100 loses info)
- Formal vocabulary differs from informal tweets
- No emoji patterns the model learned from Twitter

---

## 🛠️ Customising

Edit `models.py` to tweak:
```python
MAX_VOCAB = 20_000   # vocabulary size
MAX_LEN   = 100      # sequence length
EMBED_DIM = 100      # must match GloVe file
EPOCHS    = 10       # max epochs (EarlyStopping applies)
```
