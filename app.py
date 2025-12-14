# Import library yang dibutuhkan
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Memastikan resource NLTK sudah terdownload
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# --------------------
# KONFIGURASI HALAMAN
# --------------------
st.set_page_config(
    page_title="CineSense - AI Movie Analyst",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------
# TAMPILAN CSS
# -------------
st.markdown("""
    <style>
    .stApp {
        background-color: #141414;
        color: #FFFFFF;
    }
    
    [data-testid="stForm"] {
        border: 0px;
        padding: 0px;
        background-color: transparent;
    }
    
    h1 {
        font-family: 'Bebas Neue', sans-serif;
        color: #E50914; /* Netflix Red */
        font-size: 3.5rem !important;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }
    
    h2, h3 {
        color: #f5f5f1;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    .stTextArea textarea {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #E50914 !important;
        border-radius: 4px;
    }
    
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        background-color: #E50914 !important;
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #b20710 !important;
        transform: scale(1.02);
        color: white !important;
        border: none !important;
    }
    
    div[data-testid="stMetric"] {
        background-color: #222222;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #E50914;
    }
    
    *:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    hr {
        border-color: #404040;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# FUNGSI PREPROCESSING (Seperti di Jupyter Notebook)
# ---------------------------------------------------
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
stopwords_english = set(stopwords.words('english'))
negative_words = {"no", "not", "nor", "don", "don't", "ain", "aren", "aren't", 
                  "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", 
                  "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", 
                  "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", 
                  "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", 
                  "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"} 
stopwords_final = stopwords_english - negative_words

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    clean_tokens = []
    for word, tag in pos_tags:
        if word not in stopwords_final and len(word) > 1:
            wn_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            clean_tokens.append(lemma)        
    return ' '.join(clean_tokens)

# --------------------------
# LOAD MODEL DAN VECTORIZER
# --------------------------
@st.cache_resource
def load_assets():
    try:
        vec = pickle.load(open('source/vectorizer.pkl', 'rb'))
        mod = pickle.load(open('source/logistic_model.pkl', 'rb'))
        return vec, mod
    except Exception as e:
        return None, None

vectorizer, model = load_assets()

# ----------------------------
# HEADER DAN PENANGANAN ERROR
# ----------------------------
st.title("CINE SENSE")
st.markdown("<h3 style='text-align: center; color: #b3b3b3; margin-bottom: 30px;'>Unlock the True Sentiment of Movie Reviews</h3>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ File model atau vectorizer tidak ditemukan di folder 'source'.")
    st.stop()

# ---------------------
# INPUT DAN PREDICTION
# ---------------------
with st.form(key='sentiment_form'):
    input_text = st.text_area(
        "Masukkan Komentar Anda dalam Bahasa Inggris", 
        height=150, 
        placeholder="Example: It was an amazing movie with stunning visuals and a gripping storyline...",
        help="Tekan Ctrl+Enter untuk menjalankan analisis secara cepat."
    )
    predict_btn = st.form_submit_button("ANALYZE SENTIMENT")

if predict_btn and input_text:
    with st.spinner('Analyzing patterns...'):
        text_clean = text_preprocessing(input_text) 
        if not text_clean:
             st.warning("⚠️ Teks tidak mengandung kata yang valid untuk dianalisis.")
        else:
            vector_input = vectorizer.transform([text_clean])
            prediction = model.predict(vector_input)[0]
            probability = model.predict_proba(vector_input).max()
            st.markdown("---")
            st.markdown("<h2 style='text-align: center;'>Analysis Result</h2>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if prediction == 1:
                    st.markdown("""
                    <div style='background-color: #1b4d26; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #2ecc71;'>
                        <h1 style='color: #2ecc71; margin:0;'>POSITIVE</h1>
                        <h1 style='font-size: 80px !important; margin:0;'>👍</h1>
                        <p style='color: #d1e7dd;'>I'm agree with that!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color: #4d1b1b; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #e74c3c;'>
                        <h1 style='color: #e74c3c; margin:0;'>NEGATIVE</h1>
                        <h1 style='font-size: 80px !important; margin:0;'>👎</h1>
                        <p style='color: #f8d7da;'>Seems disappointing...</p>
                    </div>
                    """, unsafe_allow_html=True)
            with res_col2:
                st.markdown("#### Confidence Score")
                st.write(f"Model sangat yakin sebesar **{probability*100:.1f}%** dengan keputusannya.")
                st.progress(probability)
                st.markdown("#### Top Keywords Detected")
                feature_names = vectorizer.get_feature_names_out()
                dense_vector = vector_input.todense().tolist()[0]
                phrase_scores = [pair for pair in zip(range(0, len(dense_vector)), dense_vector) if pair[1] > 0]
                sorted_phrases = sorted(phrase_scores, key=lambda t: t[1] * -1)
                top_words = [feature_names[word_id] for word_id, score in sorted_phrases[:5]]
                if not top_words:
                    st.write("No specific keywords detected.")
                else:
                    tags_html = ""
                    for w in top_words:
                        tags_html += f"<span style='background-color: #333; padding: 5px 10px; margin: 2px; border-radius: 15px; border: 1px solid #555;'>{w}</span>"
                    st.markdown(tags_html, unsafe_allow_html=True)

# ------------------
# BEHIND THE SCENES
# ------------------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h2 style='color: #E50914;'>Behind The Scenes: The AI Model</h2>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Algorithm", "📊 Performance", "🧠 Confusion Matrix"])

with tab1:
    col_exp1, col_exp2 = st.columns([1, 2])
    with col_exp1:
        st.image("https://media.geeksforgeeks.org/wp-content/uploads/20251118174609019862/2.webp", caption="Sigmoid Function")
    
    with col_exp2:
        st.markdown("### Logistic Regression")
        st.write("""
        Model utama yang digunakan dalam aplikasi ini adalah **Logistic Regression**. 
        Meskipun memiliki nama "Regression", algoritma ini digunakan untuk **Klasifikasi**.
        """)
        st.info("""
        **Cara Kerja Model:**
        1. Model mengubah komentar menjadi angka menggunakan **TF-IDF**.
        2. Model menghitung nilai numerik setiap kata, misal "bad" bernilai negatif dan "good" bernilai positif.
        3. Hasil perhitungan dipetakan ke kurva **Sigmoid** yang menghasilkan probabilitas antara 0 hingga 1.
        """)

with tab2:
    st.write("Berdasarkan hasil pengujian pada ±10.000 data testing (IMDB Dataset):")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "88.0%", "Robust")
    m2.metric("Precision", "88.0%", "High Trust")
    m3.metric("Recall", "89.0%", "Sensitive")
    m4.metric("F1-Score", "88.0%", "Balanced")
    st.write("*Model ini menunjukkan performa yang sangat konsisten dengan keseimbangan yang baik antara Precision dan Recall.*")

with tab3:
    st.write("Visualisasi bagaimana model menebak benar dan salah")
    cm_data = np.array([[4324, 616], [540, 4437]]) 
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Reds', ax=ax, cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.patch.set_facecolor('#141414')
    ax.set_facecolor('#141414')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    st.pyplot(fig)

# -------
# FOOTER
# -------
st.markdown("""
<br><hr>
<p style='text-align: center; color: #555;'>
    Created for Project UAS | © 2025 CineSense AI
</p>
""", unsafe_allow_html=True)