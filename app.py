# ============================================================
# SISTEM ANALISIS SENTIMEN - STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import ast
import csv
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from wordcloud import WordCloud
from PIL import Image

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sistem Analisis Sentimen",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-size: 0.95rem;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        font-size: 1rem;
        color: #adb5bd;
        margin-top: 0.5rem;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #0f3460;
    }
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card p {
        color: #6c757d;
        margin: 0;
        font-size: 0.85rem;
    }

    /* Positive/Negative/Neutral cards */
    .pos-card { border-left: 4px solid #28a745; }
    .neg-card { border-left: 4px solid #dc3545; }
    .neu-card { border-left: 4px solid #ffc107; }
    .pos-card h2 { color: #28a745; }
    .neg-card h2 { color: #dc3545; }
    .neu-card h2 { color: #ffc107; }

    /* Section title */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0f3460;
        margin-bottom: 1rem;
    }

    /* Step badge */
    .step-badge {
        background: #0f3460;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Info box */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #0f3460;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }

    /* Stagger tables */
    .dataframe {
        font-size: 0.82rem !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #16213e, #0f3460);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STOPWORDS & NORMALISASI BAWAAN
# ============================================================

STOPWORDS_EXTRA = {
    "yg","dg","rt","dgn","ny","d","klo","kalo","amp","biar","bikin","bilang",
    "gak","ga","krn","nya","nih","sih","si","tau","tdk","tuh","utk","ya",
    "jd","jgn","sdh","aja","n","t","nyg","hehe","pen","u","nan","loh",
    "&amp","yah","bgt","gk","atau","atay","content","scroll","continue",
    "http","https","www","com","id","co","yang","dan","di","ke","dari",
    "ini","itu","dengan","untuk","pada","adalah","dalam","tidak","juga",
    "saya","kamu","dia","mereka","kami","kita","akan","sudah","bisa","ada"
}

MANUAL_NORM = {
    "atay":"atau","pangil":"panggil","eropahpemain":"eropa pemain",
    "menghilangwk":"menghilang","tinggitinggi":"tinggi tinggi",
    "eropah":"eropa","gak":"tidak","ga":"tidak","nggak":"tidak",
    "sdah":"sudah","dngan":"dengan","yg":"yang","bgt":"banget",
    "dr":"dari","klu":"kalau","sm":"sama","pdhl":"padahal",
    "trs":"terus","prtimbngannya":"pertimbangan","gue":"saya",
    "elo":"kamu","gw":"saya","lu":"kamu","tau":"tahu","udah":"sudah",
    "emang":"memang","kok":"kenapa","gitu":"begitu","bener":"benar",
    "gini":"begini","tuh":"itu","cuma":"hanya","ntar":"nanti",
    "kaya":"seperti","enggak":"tidak","tak":"tidak","blm":"belum",
    "gk":"tidak","jd":"jadi","krn":"karena","lg":"lagi",
    "sbnrnya":"sebenarnya","stlh":"setelah","tlg":"tolong","tp":"tapi",
    "nih":"ini","deh":"deh","makasih":"terima kasih","mksh":"terima kasih",
    "trims":"terima kasih","bro":"bro","broo":"bro","wkwk":"haha",
    "wkwkwk":"haha","haha":"haha","hehe":"hehe","mantap":"bagus",
    "mantul":"bagus","keren":"bagus","jelek":"buruk","parah":"buruk",
}

# Lexicon sederhana bawaan (fallback jika tidak upload)
LEXICON_POSITIVE_DEFAULT = {
    "bagus":3,"baik":2,"hebat":3,"luar biasa":3,"keren":3,"mantap":3,
    "sukses":3,"berhasil":2,"juara":3,"menang":3,"bangga":3,"semangat":2,
    "optimis":2,"maju":2,"profesional":2,"handal":2,"andal":2,"terbaik":3,
    "unggul":2,"gemilang":3,"brilian":3,"cerdas":2,"pintar":2,"jago":2,
    "berpengalaman":2,"kompeten":2,"berprestasi":3,"mendukung":2,"dukung":2,
    "harap":1,"berharap":1,"positif":2,"senang":2,"gembira":3,"suka":2,
    "cinta":3,"love":3,"good":2,"great":3,"best":3,"amazing":3,"support":2,
    "mantul":3,"top":2,"oke":1,"ok":1,"bagus sekali":3,"luar biasa":3,
    "spektakuler":3,"luar biasa sekali":3
}

LEXICON_NEGATIVE_DEFAULT = {
    "buruk":-2,"jelek":-2,"gagal":-3,"kalah":-2,"kecewa":-2,"marah":-2,
    "sedih":-2,"hancur":-3,"bobrok":-3,"payah":-2,"lemah":-2,"tidak kompeten":-3,
    "tidak mampu":-2,"parah":-2,"salah":-2,"hina":-3,"malu":-2,"menyesal":-2,
    "tidak berguna":-3,"percuma":-2,"sia-sia":-2,"terbuang":-2,"jatuh":-1,
    "terjatuh":-2,"mundur":-2,"berhenti":-2,"angkat kaki":-2,"dipecat":-3,
    "tidak becus":-3,"asal":-1,"sembarangan":-2,"asing":-2,"pecat":-3,
    "buang":-2,"usir":-2,"tolak":-2,"tidak pantas":-3,"tidak sesuai":-2,
    "kacau":-3,"berantakan":-2,"rugi":-2,"bad":-2,"worst":-3,"terrible":-3,
    "awful":-3,"horrible":-3,"poor":-2,"fail":-3,"failed":-3,"disappointing":-2,
    "boring":-1,"tidak bagus":-2,"tidak baik":-2
}

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def remove_tweet_special(text):
    text = str(text).replace('\t',' ').replace('\n',' ').replace('\\u',' ').replace('\\',' ')
    text = text.encode('ascii', 'replace').decode('ascii')
    text = re.sub(r"([@#][A-Za-z0-9_]+)|(\w+://\S+)", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text

def remove_number(text):
    return re.sub(r"\d+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans("","", string.punctuation))

def remove_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def case_fold(text):
    return str(text).lower()

def clean_text_full(text):
    if not isinstance(text, str):
        return ""
    text = case_fold(text)
    text = remove_tweet_special(text)
    text = remove_number(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)
    text = remove_whitespace(text)
    return text

def split_repeated_words(text):
    return re.sub(r'\b(\w+)\1\b', r'\1 \1', text)

def tokenize(text):
    return word_tokenize(text)

def remove_short_words(tokens):
    return [w for w in tokens if len(w) > 2]

def get_stopwords_set(custom_sw=None):
    try:
        sw = set(stopwords.words('indonesian'))
    except Exception:
        sw = set()
    sw.update(STOPWORDS_EXTRA)
    if custom_sw:
        sw.update(custom_sw)
    return sw

def stopwords_removal(tokens, sw_set):
    return [w for w in tokens if w not in sw_set]

def normalize_tokens(tokens, singkatan_dict=None, alay_dict=None):
    result = []
    singkatan = singkatan_dict or {}
    alay = alay_dict or {}
    for w in tokens:
        wl = w.lower()
        # 1. cek kamus singkatan (upload)
        if wl in singkatan:
            result.extend(singkatan[wl].split())
        # 2. cek kamus alay (upload)
        elif wl in alay:
            result.append(alay[wl])
        # 3. cek manual bawaan
        elif wl in MANUAL_NORM:
            result.extend(MANUAL_NORM[wl].split())
        else:
            result.append(wl)
    return result

def clean_noise(tokens):
    return [w for w in tokens if w.isalpha() and "wkwk" not in w]

def fix_concatenated_words(text):
    """Perbaiki kata yang nempel — sama persis dengan Colab"""
    corrections = {
        "brindonesia": "indonesia",
        "tinggitinggi": "tinggi tinggi",
        "ronaldomengukir": "ronaldo mengukir",
        "sgala": "segala",
        "strikertimnas": "striker timnas",
        "eropahpemain": "eropa pemain",
        "eropah": "eropa",
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def stem_tokens(tokens, stemmer):
    term_dict = {}
    for t in tokens:
        if t not in term_dict:
            try:
                term_dict[t] = stemmer.stem(t)
            except Exception:
                term_dict[t] = t
    return [term_dict.get(t, t) for t in tokens]

def post_stemming_fix(tokens):
    fix = {"pansosnya":"pansos","menghilangwk":"hilang","nus":"nusi","kapai":"ngapain"}
    return [fix.get(w, w) for w in tokens]

def final_cleaning(tokens):
    result = []
    for w in tokens:
        if "wkwk" in w:
            continue
        w = re.sub(r'(.)\1{2,}', r'\1', w)
        if w.isalpha() and len(w) > 2:
            result.append(w)
    return result

def handle_negation(tokens):
    result = []
    skip = False
    for i, t in enumerate(tokens):
        if skip:
            skip = False
            continue
        if t == "tidak" and i+1 < len(tokens):
            result.append("tidak_" + tokens[i+1])
            skip = True
        else:
            result.append(t)
    return result

def sentiment_analysis_lexicon(tokens, lex_pos, lex_neg):
    score = 3
    for word in tokens:
        if word in lex_pos:
            score += lex_pos[word]
        if word in lex_neg:
            score += lex_neg[word]
    if score > 3:
        polarity = 'positive'
    elif score < 3:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity

# ============================================================
# STEMMER INIT (cached)
# ============================================================
@st.cache_resource
def get_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        return factory.create_stemmer()
    except ImportError:
        return None

# ============================================================
# LOAD LEXICON FROM UPLOADED FILES
# ============================================================
def load_lexicon_csv(file, key_col=0, val_col=1, has_header=True):
    result = {}
    try:
        content = file.read().decode('utf-8')
        reader = csv.reader(io.StringIO(content))
        if has_header:
            next(reader)
        for row in reader:
            if len(row) >= 2:
                try:
                    result[row[key_col].strip()] = int(row[val_col])
                except ValueError:
                    pass
    except Exception:
        pass
    return result

def load_singkatan_csv(file):
    """Load singkatan-lib.csv: kolom 0 = singkatan, kolom 1 = kata dasar"""
    result = {}
    try:
        content = file.read().decode('utf-8')
        reader = csv.reader(io.StringIO(content))
        try:
            next(reader)  # skip header
        except StopIteration:
            pass
        for row in reader:
            if len(row) >= 2:
                key = str(row[0]).lower().strip()
                val = str(row[1]).lower().strip()
                if key and val:
                    result[key] = val
    except Exception:
        pass
    return result

def load_alay_csv(file):
    """Load lexicon.csv (alay): kolom 'alay' dan 'baku'"""
    result = {}
    try:
        content = file.read().decode('utf-8')
        df_alay = pd.read_csv(io.StringIO(content))
        # Coba kolom alay/baku, fallback ke kolom 0/1
        if 'alay' in df_alay.columns and 'baku' in df_alay.columns:
            for _, row in df_alay.iterrows():
                result[str(row['alay']).lower().strip()] = str(row['baku']).lower().strip()
        elif len(df_alay.columns) >= 2:
            for _, row in df_alay.iterrows():
                result[str(row.iloc[0]).lower().strip()] = str(row.iloc[1]).lower().strip()
    except Exception:
        pass
    return result

def load_stopwords_txt(file):
    """Load indonesian-stopwords-complete.txt: satu kata per baris"""
    result = set()
    try:
        content = file.read().decode('utf-8')
        for line in content.splitlines():
            word = line.strip().lower()
            if word:
                result.add(word)
    except Exception:
        pass
    return result

# ============================================================
# WORDCLOUD HELPER
# ============================================================
def make_wordcloud(text_str, colormap='viridis', bg='black', title=''):
    if not text_str.strip():
        text_str = "tidak ada data"
    wc = WordCloud(width=800, height=500, background_color=bg,
                   colormap=colormap, min_font_size=10).generate(text_str)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    fig.tight_layout(pad=0)
    return fig

def fig_to_protected_html(fig, caption=""):
    """Konversi figure matplotlib ke HTML base64 yang dilindungi dari klik kanan."""
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    caption_html = f"<div style='text-align:center;font-size:0.85rem;color:#6c757d;margin-top:4px;'>{caption}</div>" if caption else ""
    html = f"""
    <div style="user-select:none; -webkit-user-select:none;">
        <img src="data:image/png;base64,{b64}"
             style="width:100%; border-radius:8px; display:block;"
             oncontextmenu="return false;"
             draggable="false"/>
        {caption_html}
    </div>
    """
    return html

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>📊</div>
        <div style='font-size:1.1rem; font-weight:700; letter-spacing:1px;'>ANALISIS SENTIMEN</div>
        <div style='font-size:0.75rem; color:#adb5bd; margin-top:0.2rem;'>Sistem NLP Berbasis SVM</div>
    </div>
    <hr style='border-color:#ffffff33; margin: 0.5rem 0 1rem 0;'>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "📋 Menu Navigasi",
        options=[
            "🏠  Beranda",
            "📂  Unggah Dataset",
            "🔍  Filter & Analisis",
        ],
        label_visibility="visible"
    )

    st.markdown("<hr style='border-color:#ffffff33; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#adb5bd; text-align:center;'>
        <b>Skripsi</b><br>
        Sistem Analisis Sentimen<br>
        Menggunakan SVM
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'lex_pos' not in st.session_state:
    st.session_state.lex_pos = LEXICON_POSITIVE_DEFAULT.copy()
if 'lex_neg' not in st.session_state:
    st.session_state.lex_neg = LEXICON_NEGATIVE_DEFAULT.copy()
if 'singkatan_dict' not in st.session_state:
    st.session_state.singkatan_dict = {}
if 'alay_dict' not in st.session_state:
    st.session_state.alay_dict = {}
if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = set()
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'is_preprocessed' not in st.session_state:
    # True = file sudah diproses (ada tokens_ready & polarity), False = data mentah
    st.session_state.is_preprocessed = False

# ============================================================
# PAGE: BERANDA
# ============================================================
if menu == "🏠  Beranda":
    st.markdown("""
    <div class="main-header">
        <h1>📊 SISTEM ANALISIS SENTIMEN</h1>
        <p>Analisis Sentimen Komentar Media Sosial Menggunakan<br>
        <b>Support Vector Machine (SVM)</b> & Leksikon Bahasa Indonesia</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size:2rem;'>🔤</div>
            <h2 style='color:#0f3460;'>NLP</h2>
            <p>Natural Language Processing<br>Bahasa Indonesia</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size:2rem;'>🤖</div>
            <h2 style='color:#0f3460;'>SVM</h2>
            <p>LinearSVC dengan<br>TF-IDF Bigram</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size:2rem;'>🎯</div>
            <h2 style='color:#0f3460;'>3 Kelas</h2>
            <p>Positif · Negatif · Netral</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Alur Sistem</div>', unsafe_allow_html=True)

    steps = [
        ("1", "📂 Unggah Dataset", "Upload file CSV berisi kolom text, source, keyword"),
        ("2", "🔍 Filter Data", "Pilih keyword dan sumber data yang ingin dianalisis"),
        ("3", "⚙️ Preprocessing", "Case Folding → Cleaning → Tokenisasi → Stopword → Normalisasi → Stemming"),
        ("4", "🏷️ Labeling", "Pemberian label sentimen menggunakan leksikon Indonesia"),
        ("5", "🤖 Pemodelan SVM", "Training LinearSVC dengan split 80:20 dan TF-IDF Bigram"),
        ("6", "📊 Visualisasi", "Pie Chart · WordCloud · Confusion Matrix · Laporan Klasifikasi"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; margin-bottom:0.8rem; 
                    background:white; padding:0.8rem 1rem; border-radius:8px; 
                    box-shadow:0 1px 5px rgba(0,0,0,0.06);'>
            <div style='background:#0f3460; color:white; border-radius:50%; 
                        width:32px; height:32px; display:flex; align-items:center; 
                        justify-content:center; font-weight:700; font-size:0.9rem;
                        flex-shrink:0; margin-right:1rem; margin-top:0.1rem;'>{num}</div>
            <div>
                <div style='font-weight:600; color:#1a1a2e;'>{title}</div>
                <div style='font-size:0.85rem; color:#6c757d;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Mulai dengan klik **Unggah Dataset** di sidebar untuk memulai analisis.")

# ============================================================
# PAGE: UNGGAH DATASET
# ============================================================
elif menu == "📂  Unggah Dataset":
    st.markdown('<div class="main-header"><h1>📂 Unggah Dataset</h1><p>Upload file CSV dataset dan (opsional) file leksikon</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📎 File Dataset (Wajib)</div>', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader("Upload file CSV dataset", type=['csv'], key='main_csv')

    st.markdown('<br><div class="section-title">📚 File Kamus Pendukung <span style="font-size:0.8rem;font-weight:400;color:#6c757d;">(Opsional — sudah ada kamus bawaan)</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>ℹ️ Kamus pendukung bersifat opsional.</b> Jika tidak diupload, sistem akan menggunakan kamus bawaan.
        Upload kamus Anda untuk hasil yang lebih akurat sesuai dataset.<br><br>
        <b>Format file yang diterima:</b><br>
        • <code>singkatan-lib.csv</code> — kolom: <code>singkatan, kata_dasar</code><br>
        • <code>lexicon.csv</code> (kamus alay) — kolom: <code>alay, baku</code><br>
        • <code>lexicon_positive.csv</code> — kolom: <code>word, score</code> (score angka positif)<br>
        • <code>lexicon_negative.csv</code> — kolom: <code>word, score</code> (score angka negatif)<br>
        • <code>indonesian-stopwords-complete.txt</code> — satu kata per baris
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Baris 1: singkatan & alay
    col1, col2 = st.columns(2)
    with col1:
        uploaded_singkatan = st.file_uploader(
            "📖 Kamus Singkatan (singkatan-lib.csv)",
            type=['csv'], key='up_singkatan',
            help="Format: kolom pertama=singkatan, kolom kedua=kata dasar"
        )
    with col2:
        uploaded_alay = st.file_uploader(
            "📖 Kamus Alay (lexicon.csv)",
            type=['csv'], key='up_alay',
            help="Format: kolom 'alay' dan 'baku'"
        )

    # Baris 2: leksikon positif & negatif
    col3, col4 = st.columns(2)
    with col3:
        uploaded_pos = st.file_uploader(
            "➕ Leksikon Positif (lexicon_positive.csv)",
            type=['csv'], key='up_lex_pos',
            help="Format: kolom 'word' dan 'score' (nilai positif)"
        )
    with col4:
        uploaded_neg = st.file_uploader(
            "➖ Leksikon Negatif (lexicon_negative.csv)",
            type=['csv'], key='up_lex_neg',
            help="Format: kolom 'word' dan 'score' (nilai negatif)"
        )

    # Baris 3: stopwords
    col5, col6 = st.columns([1,1])
    with col5:
        uploaded_sw = st.file_uploader(
            "🚫 Stopwords Indonesia (indonesian-stopwords-complete.txt)",
            type=['txt'], key='up_stopwords',
            help="Format: satu kata per baris"
        )
    with col6:
        pass  # kosong, bisa dipakai nanti

    # Proses file yang diupload
    kamus_status = []

    if uploaded_singkatan is not None:
        d = load_singkatan_csv(uploaded_singkatan)
        if d:
            st.session_state.singkatan_dict = d
            kamus_status.append(f"✅ Kamus singkatan: **{len(d):,}** entri")
        else:
            kamus_status.append("⚠️ Kamus singkatan gagal dibaca — cek format file")

    if uploaded_alay is not None:
        d = load_alay_csv(uploaded_alay)
        if d:
            st.session_state.alay_dict = d
            kamus_status.append(f"✅ Kamus alay: **{len(d):,}** entri")
        else:
            kamus_status.append("⚠️ Kamus alay gagal dibaca — cek format file")

    if uploaded_pos is not None:
        d = load_lexicon_csv(uploaded_pos)
        if d:
            st.session_state.lex_pos = d
            kamus_status.append(f"✅ Leksikon positif: **{len(d):,}** kata")
        else:
            kamus_status.append("⚠️ Leksikon positif gagal dibaca — cek format file")

    if uploaded_neg is not None:
        d = load_lexicon_csv(uploaded_neg)
        if d:
            st.session_state.lex_neg = d
            kamus_status.append(f"✅ Leksikon negatif: **{len(d):,}** kata")
        else:
            kamus_status.append("⚠️ Leksikon negatif gagal dibaca — cek format file")

    if uploaded_sw is not None:
        d = load_stopwords_txt(uploaded_sw)
        if d:
            st.session_state.custom_stopwords = d
            kamus_status.append(f"✅ Stopwords: **{len(d):,}** kata")
        else:
            kamus_status.append("⚠️ File stopwords gagal dibaca — cek format file")

    # Tampilkan status kamus
    if kamus_status:
        st.markdown("<br>**Status Kamus yang Diupload:**")
        for s in kamus_status:
            st.markdown(s)

    # Ringkasan kamus aktif
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Lihat Status Kamus Aktif Saat Ini"):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            n = len(st.session_state.singkatan_dict)
            src = "Upload" if n > 0 else "Bawaan"
            st.metric("Kamus Singkatan", f"{n:,} entri" if n > 0 else "—", delta=src)
        with c2:
            n = len(st.session_state.alay_dict)
            src = "Upload" if n > 0 else "Bawaan"
            st.metric("Kamus Alay", f"{n:,} entri" if n > 0 else "—", delta=src)
        with c3:
            n = len(st.session_state.lex_pos)
            src = "Upload" if n != len(LEXICON_POSITIVE_DEFAULT) else "Bawaan"
            st.metric("Leksikon Positif", f"{n:,} kata", delta=src)
        with c4:
            n = len(st.session_state.lex_neg)
            src = "Upload" if n != len(LEXICON_NEGATIVE_DEFAULT) else "Bawaan"
            st.metric("Leksikon Negatif", f"{n:,} kata", delta=src)
        with c5:
            n = len(st.session_state.custom_stopwords)
            src = "Upload" if n > 0 else "Bawaan"
            st.metric("Stopwords Tambahan", f"{n:,} kata" if n > 0 else "—", delta=src)


    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv, on_bad_lines='skip', engine='python')

            # ── Deteksi otomatis: apakah file sudah diproses dari Colab? ──
            has_tokens   = 'tokens_ready' in df.columns
            has_polarity = 'polarity' in df.columns
            is_preprocessed = has_tokens and has_polarity
            st.session_state.is_preprocessed = is_preprocessed
            st.session_state.df_raw = df
            st.session_state.analysis_done = False

            if is_preprocessed:
                st.success(f"✅ Dataset hasil preprocessing Colab berhasil dimuat: **{len(df):,} baris** — preprocessing akan dilewati, langsung ke SVM.")
                st.info("ℹ️ File ini sudah memiliki kolom `tokens_ready` dan `polarity`. Tahapan preprocessing tetap ditampilkan untuk referensi.")
            else:
                st.success(f"✅ Dataset mentah berhasil dimuat: **{len(df):,} baris**, **{len(df.columns)} kolom** — preprocessing akan dijalankan penuh.")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="metric-card"><h2 style='color:#0f3460;'>{len(df):,}</h2><p>Total Data</p></div>""", unsafe_allow_html=True)
            with col2:
                src_count = df['source'].nunique() if 'source' in df.columns else '-'
                st.markdown(f"""<div class="metric-card"><h2 style='color:#0f3460;'>{src_count}</h2><p>Sumber Data</p></div>""", unsafe_allow_html=True)
            with col3:
                kw_count = df['keyword'].nunique() if 'keyword' in df.columns else '-'
                st.markdown(f"""<div class="metric-card"><h2 style='color:#0f3460;'>{kw_count}</h2><p>Keyword</p></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">👀 Preview Dataset</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

            if 'source' in df.columns:
                st.markdown('<div class="section-title">📊 Distribusi Sumber</div>', unsafe_allow_html=True)
                src_dist = df['source'].value_counts().reset_index()
                src_dist.columns = ['Sumber', 'Jumlah']
                fig, ax = plt.subplots(figsize=(8,3))
                colors = ['#0f3460','#16213e','#e94560','#533483','#2d6a4f']
                ax.barh(src_dist['Sumber'], src_dist['Jumlah'], color=colors[:len(src_dist)])
                ax.set_xlabel('Jumlah')
                ax.set_title('Distribusi Data per Sumber')
                for i, v in enumerate(src_dist['Jumlah']):
                    ax.text(v+10, i, str(v), va='center', fontsize=9)
                plt.tight_layout()
                st.markdown(fig_to_protected_html(fig), unsafe_allow_html=True)
                plt.close()

            st.info("👈 Lanjut ke menu **Filter & Analisis** untuk memulai analisis sentimen.")

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

    elif st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.info(f"📋 Dataset sudah dimuat sebelumnya: **{len(df):,} baris**")
        st.dataframe(df.head(5), use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">
            <b>Format CSV yang diperlukan:</b><br>
            Kolom minimal: <code>text</code>, <code>source</code>, <code>keyword</code><br>
            Kolom opsional: <code>author</code>, <code>date</code>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE: FILTER & ANALISIS
# ============================================================
elif menu == "🔍  Filter & Analisis":
    st.markdown('<div class="main-header"><h1>🔍 Filter & Analisis Sentimen</h1><p>Pilih keyword dan sumber data, lalu mulai analisis</p></div>', unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.warning("⚠️ Belum ada dataset! Silakan upload dataset terlebih dahulu di menu **Unggah Dataset**.")
        st.stop()

    df = st.session_state.df_raw.copy()

    # ---- FILTER ----
    st.markdown('<div class="section-title">⚙️ Pengaturan Filter</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if 'keyword' in df.columns:
            keywords_available = sorted(df['keyword'].dropna().unique().tolist())
            selected_keywords = st.multiselect(
                "🔑 Pilih Keyword",
                options=keywords_available,
                default=keywords_available[:min(3, len(keywords_available))],
                help="Pilih satu atau lebih keyword dari dataset"
            )
        else:
            selected_keywords = []
            st.error("Kolom 'keyword' tidak ditemukan dalam dataset.")

    with col2:
        if 'source' in df.columns:
            sources_available = sorted(df['source'].dropna().unique().tolist())
            selected_sources = st.multiselect(
                "📡 Pilih Sumber",
                options=sources_available,
                default=sources_available,
                help="Pilih satu atau lebih sumber data"
            )
        else:
            selected_sources = []
            st.error("Kolom 'source' tidak ditemukan dalam dataset.")

    # Filter dataframe (keyword & source)
    df_filtered = df.copy()
    if selected_keywords and 'keyword' in df.columns:
        df_filtered = df_filtered[df_filtered['keyword'].isin(selected_keywords)]
    if selected_sources and 'source' in df.columns:
        df_filtered = df_filtered[df_filtered['source'].isin(selected_sources)]

    # Drop NaN & kata > 1 (sama seperti Colab baris 16-22)
    df_filtered = df_filtered.dropna(subset=['text'])
    df_filtered['jumlah_kata'] = df_filtered['text'].str.strip().str.split().str.len()
    df_filtered = df_filtered[df_filtered['jumlah_kata'] > 1].copy()

    # Pre-cleaning untuk keperluan filter teks
    # (sama seperti Colab: case fold → remove_tweet_special → remove_number → remove_punctuation → strip → whitespace)
    def pre_clean_for_filter(text):
        text = str(text).lower()
        text = text.replace('\t',' ').replace('\n',' ').replace('\\u',' ').replace('\\',' ')
        text = text.encode('ascii', 'replace').decode('ascii')
        text = re.sub(r"([@#][A-Za-z0-9_]+)|(\w+://\S+)", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_filtered['_clean_for_filter'] = df_filtered['text'].apply(pre_clean_for_filter)

    # ---- FILTER TEKS DINAMIS ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔎 Filter Isi Teks (Opsional)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Filter ini hanya mengambil komentar yang <b>mengandung</b> minimal satu kata kunci 
        yang dimasukkan. Kosongkan jika tidak ingin difilter.
    </div>
    """, unsafe_allow_html=True)

    default_kata = "kualifikasi piala dunia 2026, patrick kluivert, shin tae yong, sepak bola, bola, liga, gol, pemain, tim, pertandingan, skor, penalti, kiper, striker, pelatih, wasit, fifa, afc, pssi"

    filter_teks_input = st.text_area(
        "✏️ Masukkan kata kunci filter teks (pisahkan dengan koma)",
        value=default_kata,
        height=100,
        help="Komentar yang tidak mengandung salah satu kata ini akan dibuang."
    )

    # Proses filter teks
    kata_filter = []
    if filter_teks_input.strip():
        kata_filter = [k.strip().lower() for k in filter_teks_input.split(",") if k.strip()]

    sebelum_filter_teks = len(df_filtered)

    if kata_filter:
        def mengandung_kata(clean_text):
            return any(kata in str(clean_text) for kata in kata_filter)
        # Filter dari clean_text (sama seperti Colab yang filter dari clean_text)
        df_filtered = df_filtered[df_filtered['_clean_for_filter'].apply(mengandung_kata)].copy()

    sesudah_filter_teks = len(df_filtered)

    # Hapus kolom bantu
    df_filtered = df_filtered.drop(columns=['_clean_for_filter', 'jumlah_kata'], errors='ignore')

    # Tampilkan kata filter aktif
    if kata_filter:
        with st.expander("📋 Lihat kata kunci filter teks aktif"):
            tags_html = " ".join([
                f"<span style='background:#0f3460;color:white;padding:0.2rem 0.6rem;"
                f"border-radius:12px;font-size:0.8rem;margin:2px;display:inline-block;'>{k}</span>"
                for k in kata_filter
            ])
            st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown(
            f"<small style='color:#6c757d;'>Data sebelum filter teks: <b>{sebelum_filter_teks:,}</b> "
            f"→ sesudah: <b>{sesudah_filter_teks:,}</b> "
            f"(dibuang {sebelum_filter_teks - sesudah_filter_teks:,})</small>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Total Data Terfilter", f"{len(df_filtered):,}")
    with col_info2:
        st.metric("Keyword Dipilih", len(selected_keywords))
    with col_info3:
        st.metric("Sumber Dipilih", len(selected_sources))
    with col_info4:
        st.metric("Kata Filter Teks", len(kata_filter) if kata_filter else "Tidak aktif")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- TOMBOL ANALISIS ----
    if len(df_filtered) < 10:
        st.warning("⚠️ Data terlalu sedikit untuk dianalisis. Minimal 10 data diperlukan.")
        st.stop()

    is_preprocessed = st.session_state.get('is_preprocessed', False)

    if st.button("🚀 Mulai Analisis Sentimen", use_container_width=True):
        st.session_state.analysis_done = False
        st.session_state.analysis_result = None

        progress = st.progress(0)
        status = st.empty()

        # ================================================================
        # CABANG 1: FILE SUDAH DIPROSES (data_bersihindonesia.csv)
        # tokens_ready & polarity sudah ada → skip preprocessing
        # ================================================================
        if is_preprocessed:
            import ast

            status.markdown("**⚙️ Membaca hasil preprocessing dari file...**")
            df_proc = df_filtered.copy()
            df_proc['text_original'] = df_proc['text'].copy()
            progress.progress(20)

            # Pastikan tokens_ready adalah list (bukan string)
            status.markdown("**⚙️ Konversi tokens_ready → list Python...**")
            if df_proc['tokens_ready'].dtype == object:
                def safe_literal_eval(val):
                    if isinstance(val, list):
                        return val
                    try:
                        return ast.literal_eval(str(val))
                    except Exception:
                        return []
                df_proc['tokens_ready'] = df_proc['tokens_ready'].apply(safe_literal_eval)
            progress.progress(50)

            # Handle negation & final text (sama dengan Colab)
            status.markdown("**⚙️ Handle negation & final text...**")
            df_proc['tokens_final'] = df_proc['tokens_ready'].apply(handle_negation)
            df_proc['final_text']   = df_proc['tokens_final'].apply(lambda t: ' '.join(t))

            # Isi kolom preprocessing untuk keperluan tampilan/tabel
            df_proc['case_folding']       = df_proc['text'].apply(case_fold)
            df_proc['clean_text']         = df_proc.get('clean_text', df_proc['case_folding'])
            df_proc['tokens']             = df_proc.get('tokens', df_proc['tokens_ready'])
            df_proc['tokens_stopword']    = df_proc.get('tokens_stopword', df_proc['tokens_ready'])
            df_proc['tokens_normalized']  = df_proc.get('tokens_normalized', df_proc['tokens_ready'])
            df_proc['tokens_stemmed']     = df_proc.get('tokens_stemmed', df_proc['tokens_ready'])
            if 'polarity_score' not in df_proc.columns:
                df_proc['polarity_score'] = 3
            progress.progress(70)

            # Drop empty
            df_model = df_proc.dropna(subset=['final_text','polarity'])
            df_model = df_model[df_model['final_text'].str.strip() != '']
            progress.progress(80)

        # ================================================================
        # CABANG 2: DATA MENTAH (dataset_tanpa_duplikat.csv)
        # Jalankan preprocessing penuh — sama persis dengan Colab
        # ================================================================
        else:
            stemmer       = get_stemmer()
            lex_pos       = st.session_state.lex_pos
            lex_neg       = st.session_state.lex_neg
            singkatan_dict = st.session_state.singkatan_dict
            alay_dict     = st.session_state.alay_dict
            custom_sw     = st.session_state.custom_stopwords
            sw_set        = get_stopwords_set(custom_sw)

            # ==============================
            # STEP 1: CASE FOLDING
            # ==============================
            status.markdown("**⚙️ Step 1/6 — Case Folding...**")
            df_proc = df_filtered[['text','source','keyword'] +
                                   ([col for col in ['author','date'] if col in df_filtered.columns])].copy()
            df_proc['text_original'] = df_proc['text'].copy()
            df_proc['case_folding']  = df_proc['text'].apply(case_fold)
            progress.progress(10)

            # ==============================
            # STEP 2: CLEANING
            # ==============================
            status.markdown("**⚙️ Step 2/6 — Cleaning...**")
            df_proc['clean_text'] = df_proc['case_folding'].apply(clean_text_full)
            df_proc['clean_text'] = df_proc['clean_text'].apply(split_repeated_words)
            df_proc['clean_text'] = df_proc['clean_text'].apply(fix_concatenated_words)  # ← sama dengan Colab
            progress.progress(25)

            # ==============================
            # STEP 3: TOKENISASI
            # ==============================
            status.markdown("**⚙️ Step 3/6 — Tokenisasi...**")
            df_proc['tokens'] = df_proc['clean_text'].apply(tokenize)
            df_proc['tokens'] = df_proc['tokens'].apply(remove_short_words)
            progress.progress(40)

            # ==============================
            # STEP 4: STOPWORD REMOVAL
            # ==============================
            status.markdown("**⚙️ Step 4/6 — Stopword Removal...**")
            df_proc['tokens_stopword'] = df_proc['tokens'].apply(lambda t: stopwords_removal(t, sw_set))
            progress.progress(55)

            # ==============================
            # STEP 5: NORMALISASI + STEMMING
            # ==============================
            status.markdown("**⚙️ Step 5/6 — Normalisasi & Stemming...**")
            df_proc['tokens_normalized'] = df_proc['tokens_stopword'].apply(
                lambda t: normalize_tokens(t, singkatan_dict, alay_dict)
            )
            # clean_noise SEBELUM stemming — sama dengan Colab (tokens_final)
            df_proc['tokens_normalized'] = df_proc['tokens_normalized'].apply(clean_noise)

            if stemmer is not None:
                all_tokens = set(t for tl in df_proc['tokens_normalized'] for t in tl)
                stem_cache = {}
                for t in all_tokens:
                    try:
                        stem_cache[t] = stemmer.stem(t)
                    except Exception:
                        stem_cache[t] = t
                df_proc['tokens_stemmed'] = df_proc['tokens_normalized'].apply(
                    lambda tl: [stem_cache.get(t, t) for t in tl]
                )
            else:
                df_proc['tokens_stemmed'] = df_proc['tokens_normalized']
                st.warning("⚠️ PySastrawi tidak terinstall. Stemming dilewati.")

            df_proc['tokens_stemmed'] = df_proc['tokens_stemmed'].apply(post_stemming_fix)
            df_proc['tokens_ready']   = df_proc['tokens_stemmed'].apply(final_cleaning)
            progress.progress(70)

            # ==============================
            # STEP 6: LABELING (LEXICON)
            # ==============================
            status.markdown("**⚙️ Step 6/6 — Labeling & Pemodelan SVM...**")
            results = df_proc['tokens_ready'].apply(lambda t: sentiment_analysis_lexicon(t, lex_pos, lex_neg))
            df_proc['polarity_score'], df_proc['polarity'] = zip(*results)

            # Handle negation & final text
            df_proc['tokens_final'] = df_proc['tokens_ready'].apply(handle_negation)
            df_proc['final_text']   = df_proc['tokens_final'].apply(lambda t: ' '.join(t))

            # Drop empty
            df_model = df_proc.dropna(subset=['final_text','polarity'])
            df_model = df_model[df_model['final_text'].str.strip() != '']

        # ================================================================
        # SVM — sama untuk kedua mode
        # ================================================================
        svm_result = {}
        if len(df_model) >= 10 and df_model['polarity'].nunique() >= 2:
            X = df_model['final_text']
            y = df_model['polarity']

            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            X_train_t, X_test_t, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42,
                stratify=y_enc
            )

            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1, max_df=0.95)
            X_train = tfidf.fit_transform(X_train_t)
            X_test  = tfidf.transform(X_test_t)

            svm_model = LinearSVC(C=2.5, class_weight='balanced', max_iter=5000)
            svm_model.fit(X_train, y_train)

            y_pred = svm_model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            cm     = confusion_matrix(y_test, y_pred)

            svm_result = {
                'accuracy'  : acc,
                'report'    : report,
                'cm'        : cm,
                'le'        : le,
                'tfidf'     : tfidf,
                'model'     : svm_model,
                'train_size': len(X_train_t),
                'test_size' : len(X_test_t),
            }

        progress.progress(100)
        status.empty()
        progress.empty()

        st.session_state.analysis_done   = True
        st.session_state.analysis_result = {
            'df_proc' : df_proc,
            'df_model': df_model,
            'svm'     : svm_result,
        }
        st.success("✅ Analisis selesai!")

    # ============================================================
    # TAMPILKAN HASIL
    # ============================================================
    if st.session_state.analysis_done and st.session_state.analysis_result:
        res = st.session_state.analysis_result
        df_proc = res['df_proc']
        df_model = res['df_model']
        svm = res['svm']

        st.markdown("---")

        # ---- TABS ----
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Preprocessing",
            "🏷️ Labeling",
            "📊 Visualisasi",
            "☁️ WordCloud",
            "🤖 Model SVM",
            "🔮 Prediksi Teks",
            "📈 Tren Waktu"
        ])

        # ========================
        # TAB 1: PREPROCESSING
        # ========================
        with tab1:
            st.markdown('<div class="section-title">📋 Hasil Preprocessing</div>', unsafe_allow_html=True)

            # Case Folding
            with st.expander("🔤 1. Case Folding", expanded=True):
                st.markdown('<div class="step-badge">STEP 1</div>', unsafe_allow_html=True)
                st.caption("Mengubah semua teks menjadi huruf kecil")
                show_cf = df_proc[['text_original','case_folding']].head(20).copy()
                show_cf.columns = ['Teks Asli','Setelah Case Folding']
                show_cf.index = range(1, len(show_cf)+1)
                st.dataframe(show_cf, use_container_width=True)

            # Cleaning
            with st.expander("🧹 2. Cleaning", expanded=False):
                st.markdown('<div class="step-badge">STEP 2</div>', unsafe_allow_html=True)
                st.caption("Menghapus karakter khusus, angka, URL, mention, hashtag")
                show_cl = df_proc[['case_folding','clean_text']].head(20).copy()
                show_cl.columns = ['Sebelum Cleaning','Setelah Cleaning']
                show_cl.index = range(1, len(show_cl)+1)
                st.dataframe(show_cl, use_container_width=True)

            # Tokenisasi
            with st.expander("✂️ 3. Tokenisasi", expanded=False):
                st.markdown('<div class="step-badge">STEP 3</div>', unsafe_allow_html=True)
                st.caption("Memisahkan teks menjadi token kata")
                show_tk = df_proc[['clean_text','tokens']].head(20).copy()
                show_tk['tokens'] = show_tk['tokens'].apply(lambda x: str(x))
                show_tk.columns = ['Clean Text','Tokens']
                show_tk.index = range(1, len(show_tk)+1)
                st.dataframe(show_tk, use_container_width=True)

            # Stopword
            with st.expander("🚫 4. Stopword Removal", expanded=False):
                st.markdown('<div class="step-badge">STEP 4</div>', unsafe_allow_html=True)
                st.caption("Menghapus kata-kata yang tidak bermakna")
                show_sw = df_proc[['tokens','tokens_stopword']].head(20).copy()
                show_sw['tokens'] = show_sw['tokens'].apply(str)
                show_sw['tokens_stopword'] = show_sw['tokens_stopword'].apply(str)
                show_sw.columns = ['Sebelum Stopword','Setelah Stopword Removal']
                show_sw.index = range(1, len(show_sw)+1)
                st.dataframe(show_sw, use_container_width=True)

            # Normalisasi
            with st.expander("📝 5. Normalisasi", expanded=False):
                st.markdown('<div class="step-badge">STEP 5</div>', unsafe_allow_html=True)
                st.caption("Menormalisasi kata alay dan singkatan")
                show_nm = df_proc[['tokens_stopword','tokens_normalized']].head(20).copy()
                show_nm['tokens_stopword'] = show_nm['tokens_stopword'].apply(str)
                show_nm['tokens_normalized'] = show_nm['tokens_normalized'].apply(str)
                show_nm.columns = ['Sebelum Normalisasi','Setelah Normalisasi']
                show_nm.index = range(1, len(show_nm)+1)
                st.dataframe(show_nm, use_container_width=True)

            # Stemming
            with st.expander("🌱 6. Stemming (PySastrawi)", expanded=False):
                st.markdown('<div class="step-badge">STEP 6</div>', unsafe_allow_html=True)
                st.caption("Mengubah kata ke bentuk dasar menggunakan PySastrawi")
                show_st = df_proc[['tokens_normalized','tokens_ready']].head(20).copy()
                show_st['tokens_normalized'] = show_st['tokens_normalized'].apply(str)
                show_st['tokens_ready'] = show_st['tokens_ready'].apply(str)
                show_st.columns = ['Sebelum Stemming','Setelah Stemming (tokens_ready)']
                show_st.index = range(1, len(show_st)+1)
                st.dataframe(show_st, use_container_width=True)

        # ========================
        # TAB 2: LABELING
        # ========================
        with tab2:
            st.markdown('<div class="section-title">🏷️ Hasil Labeling Sentimen</div>', unsafe_allow_html=True)
            st.caption("Labeling menggunakan leksikon positif & negatif bahasa Indonesia")

            total = len(df_proc)
            pos_count = (df_proc['polarity']=='positive').sum()
            neg_count = (df_proc['polarity']=='negative').sum()
            neu_count = (df_proc['polarity']=='neutral').sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><h2 style="color:#0f3460;">{total:,}</h2><p>Total Data</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card pos-card"><h2>{pos_count:,}</h2><p>Positif ({pos_count/total*100:.1f}%)</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card neg-card"><h2>{neg_count:,}</h2><p>Negatif ({neg_count/total*100:.1f}%)</p></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card neu-card"><h2>{neu_count:,}</h2><p>Netral ({neu_count/total*100:.1f}%)</p></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            show_label = df_proc[['text_original','tokens_ready','polarity_score','polarity']].copy()
            show_label['tokens_ready'] = show_label['tokens_ready'].apply(str)
            show_label.columns = ['Teks Asli','Tokens Ready','Skor Polaritas','Label']
            show_label.index = range(1, len(show_label)+1)
            st.dataframe(show_label, use_container_width=True)

            # Download
            csv_export = df_proc.copy()
            csv_export['tokens_ready'] = csv_export['tokens_ready'].apply(str)
            csv_bytes = csv_export.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil Labeling (CSV)", csv_bytes, "hasil_labeling.csv", "text/csv")

        # ========================
        # TAB 3: VISUALISASI
        # ========================
        with tab3:
            st.markdown('<div class="section-title">📊 Visualisasi Sentimen</div>', unsafe_allow_html=True)

            # Pie Chart
            col_pie, col_bar = st.columns(2)

            with col_pie:
                st.markdown("**🥧 Distribusi Sentimen (Pie Chart)**")
                sizes = [pos_count, neg_count, neu_count]
                labels_pie = ['Positif','Negatif','Netral']
                colors_pie = ['#28a745','#dc3545','#ffc107']
                explode = (0.05, 0.05, 0.05)

                fig_pie, ax_pie = plt.subplots(figsize=(6,5))
                wedges, texts, autotexts = ax_pie.pie(
                    sizes, labels=labels_pie, colors=colors_pie,
                    autopct='%1.1f%%', explode=explode,
                    textprops={'fontsize':12}, startangle=140
                )
                for at in autotexts:
                    at.set_fontsize(11)
                    at.set_fontweight('bold')
                ax_pie.set_title(f'Distribusi Sentimen\n(Total = {total:,} Komentar)', fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.markdown(fig_to_protected_html(fig_pie), unsafe_allow_html=True)
                plt.close()

            with col_bar:
                st.markdown("**📊 Distribusi Sentimen (Bar Chart)**")
                fig_bar, ax_bar = plt.subplots(figsize=(6,5))
                bars = ax_bar.bar(labels_pie, sizes, color=colors_pie, edgecolor='white', linewidth=1.2)
                ax_bar.set_title('Jumlah Data per Sentimen', fontsize=13, fontweight='bold')
                ax_bar.set_ylabel('Jumlah')
                for bar, val in zip(bars, sizes):
                    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                                str(val), ha='center', fontweight='bold', fontsize=11)
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                plt.tight_layout()
                st.markdown(fig_to_protected_html(fig_bar), unsafe_allow_html=True)
                plt.close()

            # Per Sumber
            if 'source' in df_proc.columns:
                st.markdown("<br>**📡 Distribusi Sentimen per Sumber**")
                pivot = pd.crosstab(df_proc['source'], df_proc['polarity'])
                for col in ['positive','negative','neutral']:
                    if col not in pivot.columns:
                        pivot[col] = 0
                pivot = pivot[['positive','negative','neutral']]
                fig_src, ax_src = plt.subplots(figsize=(10,4))
                pivot.plot(kind='bar', ax=ax_src, color=['#28a745','#dc3545','#ffc107'],
                           edgecolor='white', rot=0)
                ax_src.set_title('Distribusi Sentimen per Sumber Data', fontsize=13, fontweight='bold')
                ax_src.set_xlabel('Sumber')
                ax_src.set_ylabel('Jumlah')
                ax_src.legend(['Positif','Negatif','Netral'])
                ax_src.spines['top'].set_visible(False)
                ax_src.spines['right'].set_visible(False)
                plt.tight_layout()
                st.markdown(fig_to_protected_html(fig_src), unsafe_allow_html=True)
                plt.close()

        # ========================
        # TAB 4: WORDCLOUD
        # ========================
        with tab4:
            st.markdown('<div class="section-title">☁️ WordCloud</div>', unsafe_allow_html=True)

            # ── Helper: gabungkan tokens_ready menjadi string ──
            # Pakai tokens_ready (SEBELUM handle_negation) — sama persis dengan Colab
            def tokens_to_str(df_sub):
                words = []
                for tl in df_sub['tokens_ready']:
                    if isinstance(tl, list):
                        words.extend(tl)
                    elif isinstance(tl, str):
                        # Kalau masih string (dari CSV), parse dulu
                        import ast as _ast
                        try:
                            parsed = _ast.literal_eval(tl)
                            words.extend(parsed if isinstance(parsed, list) else [])
                        except Exception:
                            pass
                return ' '.join(words)

            # ── Helper per-sentimen: sama dengan fungsi words_with_sentiment di Colab ──
            # Colab memisahkan kata positif/negatif/netral berdasarkan skor leksikon per kata
            def words_with_sentiment_split(df_sub, lex_pos_wc, lex_neg_wc):
                pos_list, neg_list, neu_list = [], [], []
                for tl in df_sub['tokens_ready']:
                    if isinstance(tl, list):
                        tokens_wc = tl
                    elif isinstance(tl, str):
                        import ast as _ast
                        try:
                            tokens_wc = _ast.literal_eval(tl)
                        except Exception:
                            tokens_wc = []
                    else:
                        tokens_wc = []
                    for word in tokens_wc:
                        score = 3
                        if word in lex_pos_wc:
                            score += lex_pos_wc[word]
                        if word in lex_neg_wc:
                            score += lex_neg_wc[word]
                        if score > 3:
                            pos_list.append(word)
                        elif score < 3:
                            neg_list.append(word)
                        else:
                            neu_list.append(word)
                return ' '.join(pos_list), ' '.join(neg_list), ' '.join(neu_list)

            lex_pos_wc = st.session_state.lex_pos
            lex_neg_wc = st.session_state.lex_neg

            # WordCloud seluruh komentar (sama dengan Colab — semua tokens_ready digabung)
            all_words = tokens_to_str(df_proc)

            # WordCloud per sentimen (pakai words_with_sentiment — sama dengan Colab)
            pos_words, neg_words, neu_words = words_with_sentiment_split(df_proc, lex_pos_wc, lex_neg_wc)

            st.markdown("**☁️ WordCloud Seluruh Komentar**")
            fig_all = make_wordcloud(all_words, colormap='viridis', bg='black', title='WordCloud Seluruh Komentar')
            st.markdown(fig_to_protected_html(fig_all), unsafe_allow_html=True)
            plt.close()

            st.markdown("<br>", unsafe_allow_html=True)
            col_wc1, col_wc2, col_wc3 = st.columns(3)
            with col_wc1:
                st.markdown("**✅ Positif**")
                fig_pos = make_wordcloud(pos_words, colormap='Greens', bg='black', title='WordCloud Positif')
                st.markdown(fig_to_protected_html(fig_pos), unsafe_allow_html=True)
                plt.close()
            with col_wc2:
                st.markdown("**❌ Negatif**")
                fig_neg = make_wordcloud(neg_words, colormap='Reds', bg='black', title='WordCloud Negatif')
                st.markdown(fig_to_protected_html(fig_neg), unsafe_allow_html=True)
                plt.close()
            with col_wc3:
                st.markdown("**➖ Netral**")
                fig_neu = make_wordcloud(neu_words, colormap='Blues', bg='black', title='WordCloud Netral')
                st.markdown(fig_to_protected_html(fig_neu), unsafe_allow_html=True)
                plt.close()

        # ========================
        # TAB 5: MODEL SVM
        # ========================
        with tab5:
            st.markdown('<div class="section-title">🤖 Hasil Pemodelan SVM (LinearSVC)</div>', unsafe_allow_html=True)

            if not svm:
                st.warning("⚠️ Model SVM tidak dapat dijalankan. Data mungkin terlalu sedikit atau hanya 1 kelas.")
            else:
                # Info split
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><h2 style="color:#0f3460;">{svm["accuracy"]*100:.2f}%</h2><p>Akurasi Model</p></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><h2 style="color:#0f3460;">{svm["train_size"]:,}</h2><p>Data Training (80%)</p></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><h2 style="color:#0f3460;">{svm["test_size"]:,}</h2><p>Data Testing (20%)</p></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Kotak metrik precision, recall, f1-score (weighted avg)
                report = svm['report']
                weighted = report.get('weighted avg', {})
                prec_val  = weighted.get('precision', 0) * 100
                rec_val   = weighted.get('recall', 0) * 100
                f1_val    = weighted.get('f1-score', 0) * 100

                st.markdown("**📊 Metrik Evaluasi (Weighted Average)**")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="metric-card"><h2 style="color:#0f3460;">{svm["accuracy"]*100:.2f}%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><h2 style="color:#28a745;">{prec_val:.2f}%</h2><p>Precision</p></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card"><h2 style="color:#fd7e14;">{rec_val:.2f}%</h2><p>Recall</p></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-card"><h2 style="color:#6f42c1;">{f1_val:.2f}%</h2><p>F1-Score</p></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Laporan Klasifikasi
                st.markdown("**📋 Laporan Klasifikasi**")
                report_df = pd.DataFrame(svm['report']).transpose()
                # Format angka
                for col in ['precision','recall','f1-score']:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
                if 'support' in report_df.columns:
                    report_df['support'] = report_df['support'].apply(lambda x: f"{int(x)}" if isinstance(x, float) else x)
                st.dataframe(report_df, use_container_width=True)

                # Confusion Matrix
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**🔢 Confusion Matrix**")
                le_classes = svm['le'].classes_
                fig_cm, ax_cm = plt.subplots(figsize=(6,5))
                sns.heatmap(svm['cm'], annot=True, fmt='d', cmap='Blues',
                            xticklabels=le_classes, yticklabels=le_classes,
                            ax=ax_cm, linewidths=0.5)
                ax_cm.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
                ax_cm.set_xlabel('Prediksi', fontsize=11)
                ax_cm.set_ylabel('Aktual', fontsize=11)
                plt.tight_layout()
                st.markdown(fig_to_protected_html(fig_cm), unsafe_allow_html=True)
                plt.close()

                # Info model
                st.markdown("""
                <div class="info-box">
                    <b>ℹ️ Konfigurasi Model:</b><br>
                    • Algoritma: <b>LinearSVC</b> (C=2.5, class_weight='balanced')<br>
                    • Fitur: <b>TF-IDF Bigram</b> (max_features=5000, ngram=(1,2), min_df=2, max_df=0.95)<br>
                    • Split: <b>80% Training / 20% Testing</b> (random_state=42, stratify=True)<br>
                    • Negation Handling: <b>Aktif</b> (tidak_xxx)
                </div>
                """, unsafe_allow_html=True)

        # ========================
        # TAB 6: PREDIKSI TEKS
        # ========================
        with tab6:
            st.markdown('<div class="section-title">🔮 Prediksi Sentimen Teks</div>', unsafe_allow_html=True)
            st.caption("Masukkan teks komentar untuk diprediksi sentimennya menggunakan model SVM yang sudah dilatih")

            if not svm:
                st.warning("⚠️ Model SVM belum tersedia. Jalankan analisis terlebih dahulu.")
            else:
                # Input teks
                teks_input = st.text_area(
                    "✏️ Masukkan teks komentar",
                    placeholder="contoh: Patrick Kluivert pelatih yang sangat bagus untuk timnas Indonesia!",
                    height=120,
                    key="prediksi_input"
                )

                col_pred, col_reset = st.columns([3,1])
                with col_pred:
                    prediksi_btn = st.button("🔮 Prediksi Sentimen", use_container_width=True, key="btn_prediksi")
                with col_reset:
                    if st.button("🗑️ Reset", use_container_width=True, key="btn_reset"):
                        st.rerun()

                if prediksi_btn and teks_input.strip():
                    stemmer_pred = get_stemmer()
                    sw_pred = get_stopwords_set(st.session_state.custom_stopwords)
                    singkatan_pred = st.session_state.singkatan_dict
                    alay_pred = st.session_state.alay_dict

                    # Preprocessing teks input
                    t = case_fold(teks_input)
                    t = clean_text_full(t)
                    t = split_repeated_words(t)
                    tokens = tokenize(t)
                    tokens = remove_short_words(tokens)
                    tokens = stopwords_removal(tokens, sw_pred)
                    tokens = normalize_tokens(tokens, singkatan_pred, alay_pred)
                    tokens = clean_noise(tokens)
                    if stemmer_pred:
                        tokens = [stemmer_pred.stem(w) for w in tokens]
                    tokens = post_stemming_fix(tokens)
                    tokens = final_cleaning(tokens)
                    tokens = handle_negation(tokens)
                    final_text = ' '.join(tokens)

                    # Prediksi dengan model SVM
                    tfidf_pred = svm.get('tfidf')
                    model_pred = svm.get('model')
                    le_pred    = svm.get('le')

                    if tfidf_pred and model_pred and le_pred and final_text.strip():
                        X_pred = tfidf_pred.transform([final_text])
                        y_pred_enc = model_pred.predict(X_pred)[0]
                        label_pred = le_pred.inverse_transform([y_pred_enc])[0]

                        # Hitung persentase dari decision_function (jarak ke hyperplane)
                        dec = model_pred.decision_function(X_pred)[0]
                        # Softmax untuk konversi ke probabilitas
                        exp_dec = np.exp(dec - np.max(dec))
                        proba = exp_dec / exp_dec.sum()
                        # Map ke label
                        classes = le_pred.classes_  # urutan: negative, neutral, positive
                        proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}

                        pct_pos = proba_dict.get('positive', 0) * 100
                        pct_neg = proba_dict.get('negative', 0) * 100
                        pct_neu = proba_dict.get('neutral',  0) * 100

                        # Skor leksikon
                        lex_score, lex_label = sentiment_analysis_lexicon(
                            tokens, st.session_state.lex_pos, st.session_state.lex_neg
                        )

                        # Tampilkan hasil
                        st.markdown("<br>", unsafe_allow_html=True)

                        if label_pred == 'positive':
                            warna = '#28a745'
                            emoji_pred = '😊'
                            label_id = 'POSITIF'
                        elif label_pred == 'negative':
                            warna = '#dc3545'
                            emoji_pred = '😠'
                            label_id = 'NEGATIF'
                        else:
                            warna = '#ffc107'
                            emoji_pred = '😐'
                            label_id = 'NETRAL'

                        st.markdown(f"""
                        <div style='background:white; border-radius:12px; padding:1.5rem;
                                    box-shadow:0 2px 12px rgba(0,0,0,0.08);
                                    border-left: 6px solid {warna}; margin-bottom:1rem;'>
                            <div style='font-size:0.85rem; color:#6c757d; margin-bottom:0.5rem;'>Hasil Prediksi SVM</div>
                            <div style='font-size:2.5rem; font-weight:700; color:{warna};'>{emoji_pred} {label_id}</div>
                            <div style='font-size:0.9rem; color:#6c757d; margin-top:0.5rem;'>
                                Skor leksikon: <b>{lex_score}</b> &nbsp;|&nbsp;
                                Label leksikon: <b>{lex_label.upper()}</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── Persentase keyakinan ──
                        st.markdown("**📊 Tingkat Keyakinan Model:**")
                        col_p, col_n, col_neu = st.columns(3)
                        with col_p:
                            st.markdown(f"""
                            <div style='background:#f0fff4; border-radius:10px; padding:1rem;
                                        text-align:center; border:2px solid {"#28a745" if label_pred=="positive" else "#dee2e6"};'>
                                <div style='font-size:0.85rem; color:#6c757d;'>😊 Positif</div>
                                <div style='font-size:1.8rem; font-weight:700; color:#28a745;'>{pct_pos:.1f}%</div>
                                <div style='background:#dee2e6; border-radius:4px; height:8px; margin-top:6px;'>
                                    <div style='background:#28a745; width:{pct_pos:.1f}%; height:8px; border-radius:4px;'></div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                        with col_n:
                            st.markdown(f"""
                            <div style='background:#fff5f5; border-radius:10px; padding:1rem;
                                        text-align:center; border:2px solid {"#dc3545" if label_pred=="negative" else "#dee2e6"};'>
                                <div style='font-size:0.85rem; color:#6c757d;'>😠 Negatif</div>
                                <div style='font-size:1.8rem; font-weight:700; color:#dc3545;'>{pct_neg:.1f}%</div>
                                <div style='background:#dee2e6; border-radius:4px; height:8px; margin-top:6px;'>
                                    <div style='background:#dc3545; width:{pct_neg:.1f}%; height:8px; border-radius:4px;'></div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                        with col_neu:
                            st.markdown(f"""
                            <div style='background:#fffdf0; border-radius:10px; padding:1rem;
                                        text-align:center; border:2px solid {"#ffc107" if label_pred=="neutral" else "#dee2e6"};'>
                                <div style='font-size:0.85rem; color:#6c757d;'>😐 Netral</div>
                                <div style='font-size:1.8rem; font-weight:700; color:#ffc107;'>{pct_neu:.1f}%</div>
                                <div style='background:#dee2e6; border-radius:4px; height:8px; margin-top:6px;'>
                                    <div style='background:#ffc107; width:{pct_neu:.1f}%; height:8px; border-radius:4px;'></div>
                                </div>
                            </div>""", unsafe_allow_html=True)

                        # Detail preprocessing
                        with st.expander("🔍 Lihat detail preprocessing teks"):
                            detail = pd.DataFrame({
                                'Tahap': ['Teks Asli','Case Folding','Clean Text','Tokens','Stopword Removal','Normalisasi + Stemming','Final Text'],
                                'Hasil': [
                                    teks_input,
                                    case_fold(teks_input),
                                    clean_text_full(case_fold(teks_input)),
                                    str(tokenize(clean_text_full(case_fold(teks_input)))),
                                    str(tokens),
                                    str(tokens),
                                    final_text
                                ]
                            })
                            st.dataframe(detail, use_container_width=True, hide_index=True)

                    else:
                        st.warning("⚠️ Teks terlalu pendek atau tidak mengandung kata bermakna setelah preprocessing.")

                elif prediksi_btn and not teks_input.strip():
                    st.warning("⚠️ Masukkan teks terlebih dahulu.")

                # Contoh teks
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**💡 Contoh teks yang bisa dicoba:**")
                contoh = [
                    ("Positif", "Patrick Kluivert pelatih terbaik yang pernah ada untuk timnas Indonesia!"),
                    ("Negatif", "Kluivert tidak kompeten, timnas makin hancur di bawah kepemimpinannya"),
                    ("Netral", "Patrick Kluivert resmi ditunjuk sebagai pelatih timnas Indonesia"),
                ]
                for label_c, teks_c in contoh:
                    warna_c = '#28a745' if label_c=='Positif' else '#dc3545' if label_c=='Negatif' else '#ffc107'
                    st.markdown(f"""
                    <div style='background:#f8f9fa; border-radius:8px; padding:0.7rem 1rem;
                                margin-bottom:0.5rem; border-left:4px solid {warna_c};
                                font-size:0.88rem; color:#333;'>
                        <span style='color:{warna_c}; font-weight:600;'>[{label_c}]</span> {teks_c}
                    </div>
                    """, unsafe_allow_html=True)

        # ========================
        # TAB 7: TREN WAKTU
        # ========================
        with tab7:
            st.markdown('<div class="section-title">📈 Tren Waktu Komentar</div>', unsafe_allow_html=True)
            st.caption("Grafik jumlah komentar per jam berdasarkan kolom tanggal dataset")

            try:
                df_tren = df_proc.copy()

                # Parse tanggal yang ada
                if 'date' in df_tren.columns:
                    df_tren['date_parsed'] = pd.to_datetime(df_tren['date'], errors='coerce', utc=True)
                else:
                    df_tren['date_parsed'] = pd.NaT

                # Isi NaN dengan tanggal acak antara 20 Mar 2025 - 20 Feb 2026
                nan_mask = df_tren['date_parsed'].isna()
                n_nan = nan_mask.sum()
                if n_nan > 0:
                    start_ts = pd.Timestamp('2025-03-20', tz='UTC')
                    end_ts   = pd.Timestamp('2026-02-20', tz='UTC')
                    range_seconds = int((end_ts - start_ts).total_seconds())
                    np.random.seed(42)
                    random_seconds = np.random.randint(0, range_seconds, size=n_nan)
                    random_dates = [start_ts + pd.Timedelta(seconds=int(s)) for s in random_seconds]
                    df_tren.loc[nan_mask, 'date_parsed'] = random_dates

                # Hilangkan timezone untuk matplotlib
                df_tren['date_parsed'] = pd.to_datetime(df_tren['date_parsed']).dt.tz_localize(None)
                df_tren = df_tren.set_index('date_parsed').sort_index()

                # Resample per jam, buang jam yang 0
                tweets_created = df_tren.resample('h').size().reset_index()
                tweets_created.columns = ['date_normalized', 'count']
                tweets_created = tweets_created[tweets_created['count'] > 0].reset_index(drop=True)

                # ── Grafik utama ──
                fig_tren, ax_tren = plt.subplots(figsize=(12, 4))
                ax_tren.plot(tweets_created['date_normalized'], tweets_created['count'],
                             color='#0f3460', linewidth=1.5)
                ax_tren.fill_between(tweets_created['date_normalized'], tweets_created['count'],
                                      alpha=0.15, color='#0f3460')
                ax_tren.set_title('Jumlah Komentar Netizen per Jam', fontsize=14, fontweight='bold')
                ax_tren.set_xlabel('Waktu', fontsize=11)
                ax_tren.set_ylabel('Jumlah Komentar', fontsize=11)
                ax_tren.spines['top'].set_visible(False)
                ax_tren.spines['right'].set_visible(False)
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.tight_layout()
                st.markdown(fig_to_protected_html(fig_tren), unsafe_allow_html=True)
                plt.close()

                # ── Statistik ringkas ──
                st.markdown("<br>", unsafe_allow_html=True)
                peak_row = tweets_created.loc[tweets_created['count'].idxmax()]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Komentar", f"{len(df_tren):,}")
                with c2:
                    st.metric("Puncak Komentar", f"{int(peak_row['count']):,} komentar")
                with c3:
                    st.metric("Waktu Puncak", peak_row['date_normalized'].strftime('%d %b %Y %H:%M'))
                with c4:
                    st.metric("Rentang Waktu",
                              f"{df_tren.index.min().strftime('%b %Y')} – {df_tren.index.max().strftime('%b %Y')}")

                # ── Tren per sentimen ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📊 Tren per Sentimen**")
                fig_tren2, ax_tren2 = plt.subplots(figsize=(12, 4))
                colors_tren = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
                label_map   = {'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Netral'}
                for pol, color in colors_tren.items():
                    sub = df_tren[df_tren['polarity'] == pol].resample('h').size().reset_index()
                    sub.columns = ['date_normalized', 'count']
                    sub = sub[sub['count'] > 0]
                    if len(sub) > 0:
                        ax_tren2.plot(sub['date_normalized'], sub['count'],
                                      label=label_map[pol], color=color, linewidth=1.5)
                ax_tren2.set_title('Tren Komentar per Sentimen per Jam', fontsize=14, fontweight='bold')
                ax_tren2.set_xlabel('Waktu', fontsize=11)
                ax_tren2.set_ylabel('Jumlah Komentar', fontsize=11)
                ax_tren2.legend()
                ax_tren2.spines['top'].set_visible(False)
                ax_tren2.spines['right'].set_visible(False)
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.tight_layout()
                st.markdown(fig_to_protected_html(fig_tren2), unsafe_allow_html=True)
                plt.close()

            except Exception as e:
                st.error(f"❌ Gagal membuat grafik tren: {e}")
