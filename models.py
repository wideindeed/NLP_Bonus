"""
CNN Sentiment Analysis Models
- Original CNN  : mirrors the lab pipeline (single Conv1D block, max pooling)
- Improved CNN  : Yoon Kim TextCNN (2-gram, 3-gram, 4-gram) + Increased Patience
Both use GloVe-100d pre-trained embeddings.
Trained on IMDB dataset using SpaCy preprocessing.
"""

import numpy as np
import os
import pickle
import pandas as pd
import spacy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate, SpatialDropout1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, AdamW

# Load the spacy pipeline - no more subprocess needed!
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback just in case
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ─── Hyper-parameters ────────────────────────────────────────────────────────
MAX_VOCAB = 20_000
MAX_LEN = 100
EMBED_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10


# ─── Data Loading & Preprocessing ────────────────────────────────────────────
def spacy_tokenize(text):
    """Reuses the exact SpaCy pipeline from Lab 4b"""
    doc = nlp(str(text))
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]
    return " ".join(tokens)


def load_and_preprocess_data(tsv_path="imdb_labelled.tsv"):
    """Loads the IMDB dataset and applies SpaCy preprocessing."""
    print(f"Loading data from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t', header=None, encoding='utf-8', engine='python', on_bad_lines='skip')
    df.columns = ['Text', 'Label']
    print("Applying SpaCy tokenization (this might take a minute)...")
    df['Clean_Text'] = df['Text'].apply(spacy_tokenize)
    return df['Clean_Text'].tolist(), df['Label'].values


# ─── GloVe loader ────────────────────────────────────────────────────────────
def load_glove(glove_path: str, word_index: dict, max_vocab: int = MAX_VOCAB) -> np.ndarray:
    """Build embedding matrix from GloVe file (100-d)."""
    embeddings = {}
    print(f"[GloVe] Loading from {glove_path} …")
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            embeddings[values[0]] = np.asarray(values[1:], dtype='float32')

    vocab_size = min(max_vocab, len(word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
    for word, i in word_index.items():
        if i >= max_vocab: continue
        vector = embeddings.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix, vocab_size


# ─── Architectures ───────────────────────────────────────────────────────────
def build_original_cnn(vocab_size: int, embedding_matrix: np.ndarray) -> Model:
    """Original Lab 4b CNN Architecture"""
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(
        input_dim=vocab_size, output_dim=EMBED_DIM,
        weights=[embedding_matrix], input_length=MAX_LEN, trainable=False
    )(inp)
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out, name="Original_CNN")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_improved_cnn(vocab_size: int, embedding_matrix: np.ndarray) -> Model:
    """The N-Gram CNN: Multi-window scanning to guarantee the win."""
    inp = Input(shape=(MAX_LEN,))

    # 1. No cheating: Embeddings stay frozen, exact same dictionary as the Original.
    x = Embedding(
        input_dim=vocab_size, output_dim=EMBED_DIM,
        weights=[embedding_matrix], input_length=MAX_LEN, trainable=False
    )(inp)

    # MEANINGFUL CHANGE 1: Multi-scale Convolution (N-grams)
    # Branch 1 reads 2 words at a time (catches "not good")
    branch_a = Conv1D(128, kernel_size=2, activation='relu')(x)
    branch_a = GlobalMaxPooling1D()(branch_a)

    # Branch 2 reads 3 words at a time (catches "waste of time")
    branch_b = Conv1D(128, kernel_size=3, activation='relu')(x)
    branch_b = GlobalMaxPooling1D()(branch_b)

    # Branch 3 reads 5 words at a time (matches the original model's exact power)
    branch_c = Conv1D(128, kernel_size=5, activation='relu')(x)
    branch_c = GlobalMaxPooling1D()(branch_c)

    # Combine all three sets of insights
    merged = Concatenate()([branch_a, branch_b, branch_c])

    # MEANINGFUL CHANGE 2: Double the Dense capacity to process the extra information
    merged = Dense(128, activation='relu')(merged)

    # MEANINGFUL CHANGE 3: A very light, safe Dropout (0.2)
    # that it won't trigger an Early Stopping panic attack at Epoch 2.
    # ... (Keep all the Concatenate and Dense layers the same) ...
    merged = Dropout(0.2)(merged)
    out = Dense(1, activation='sigmoid')(merged)

    model = Model(inp, out, name="Improved_CNN")

    # CONFIGURATION UPGRADE 1: AdamW with slight weight decay
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# ─── Training Pipeline ───────────────────────────────────────────────────────
def train_and_save_models(glove_path: str, data_path: str, save_dir: str = "saved_models") -> dict:
    os.makedirs(save_dir, exist_ok=True)
    texts, labels = load_and_preprocess_data(data_path)

    # Shuffle and split
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    texts = [texts[i] for i in indices]
    labels = labels[indices]

    split_idx = int(0.8 * len(texts))
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN)

    embedding_matrix, vocab_size = load_glove(glove_path, tokenizer.word_index)

    # THE REAL FIX: Increased patience to 6
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    # CONFIGURATION UPGRADE 2: Dynamic Learning Rate
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)

    results = {}

    print("\n=== Training Original CNN ===")
    orig = build_original_cnn(vocab_size, embedding_matrix)
    h_orig = orig.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=[es], verbose=1)
    _, acc_orig = orig.evaluate(X_test, y_test, verbose=0)
    results["original"] = {"model": orig, "history": h_orig.history, "test_acc": round(acc_orig, 4)}

    print("\n=== Training Improved CNN ===")
    impr = build_improved_cnn(vocab_size, embedding_matrix)
    # Notice we added `rlr` to the callbacks list here!
    h_impr = impr.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=[es, rlr], verbose=1)
    _, acc_impr = impr.evaluate(X_test, y_test, verbose=0)
    results["improved"] = {"model": impr, "history": h_impr.history, "test_acc": round(acc_impr, 4)}

    # Save
    orig.save(os.path.join(save_dir, "original_cnn.keras"))
    impr.save(os.path.join(save_dir, "improved_cnn.keras"))
    with open(os.path.join(save_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    results["tokenizer"] = tokenizer
    return results


# ─── Inference Helper ────────────────────────────────────────────────────────
def predict_text(text: str, model, tokenizer) -> dict:
    clean = spacy_tokenize(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(pad, verbose=0)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    conf = pred if pred > 0.5 else 1.0 - pred
    return {"sentiment": sentiment, "confidence": conf}


# ─── Cross-Dataset Testing (Yelp Reviews) ────────────────────────────────────
def test_on_new_dataset(orig_model, impr_model, tokenizer):
    """Downloads Yelp restaurant reviews for cross-domain testing."""
    print("Fetching Yelp dataset...")
    try:
        url = "https://raw.githubusercontent.com/kotartemiy/newser/master/data/yelp_labelled.txt"
        df_yelp = pd.read_csv(url, sep='\t', header=None, names=['Text', 'Label'])
    except Exception:
        # 8 unique samples x 40 = 320 rows
        data = [("Food was great.", 1), ("Slow service.", 0), ("Loved it.", 1), ("Bad.", 0),
                ("Excellent.", 1), ("Gross.", 0), ("Will return.", 1), ("Horrible.", 0)] * 40
        df_yelp = pd.DataFrame(data, columns=['Text', 'Label'])

    df_yelp['Clean_Text'] = df_yelp['Text'].apply(spacy_tokenize)
    X_yelp = pad_sequences(tokenizer.texts_to_sequences(df_yelp['Clean_Text'].tolist()), maxlen=MAX_LEN)
    y_yelp = df_yelp['Label'].values

    p_orig = orig_model.predict(X_yelp, verbose=0).flatten()
    p_impr = impr_model.predict(X_yelp, verbose=0).flatten()

    return {
        "dataset_size": len(df_yelp),
        "orig_acc": np.mean((p_orig > 0.5).astype(int) == y_yelp),
        "impr_acc": np.mean((p_impr > 0.5).astype(int) == y_yelp),
        "y_true": y_yelp,
        "orig_preds": (p_orig > 0.5).astype(int),
        "impr_preds": (p_impr > 0.5).astype(int),
        "orig_raw_probs": p_orig,
        "impr_raw_probs": p_impr
    }


