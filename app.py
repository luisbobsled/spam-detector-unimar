import streamlit as st
import numpy as np
import joblib
import os

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Spam",
    page_icon="🛡️",
    layout="wide",
)

# ── Estilo visual ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    .titulo { font-size: 2.4rem; font-weight: 800; color: #ffffff; letter-spacing: -1px; }
    .subtitulo { font-size: 1rem; color: #8892a4; margin-bottom: 2rem; }
    .secao { font-size: 0.75rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
             color: #5c6bc0; border-bottom: 1px solid #1e2130; padding-bottom: 6px; margin: 24px 0 16px 0; }
    .resultado-spam {
        background: linear-gradient(135deg, #3b0a0a, #5c1a1a);
        border: 1px solid #c0392b; border-radius: 12px;
        padding: 28px 32px; text-align: center; margin-top: 24px;
    }
    .resultado-ok {
        background: linear-gradient(135deg, #0a2e1a, #0d3d22);
        border: 1px solid #27ae60; border-radius: 12px;
        padding: 28px 32px; text-align: center; margin-top: 24px;
    }
    .resultado-titulo { font-size: 2rem; font-weight: 800; margin-bottom: 8px; }
    .resultado-prob { font-size: 1rem; color: #aab0bc; margin-top: 8px; }
    .badge {
        display: inline-block; font-size: 0.7rem; font-weight: 700;
        padding: 3px 10px; border-radius: 99px; margin-bottom: 8px;
    }
    .badge-spam { background: #c0392b22; color: #e74c3c; border: 1px solid #c0392b; }
    .badge-ok   { background: #27ae6022; color: #2ecc71; border: 1px solid #27ae60; }
    div[data-testid="stNumberInput"] label { font-size: 0.78rem !important; color: #9ca3b0 !important; }
    div[data-testid="stSlider"] label { font-size: 0.78rem !important; color: #9ca3b0 !important; }
    .stButton > button {
        width: 100%; background: #3949ab; color: white; font-weight: 700;
        font-size: 1rem; border-radius: 8px; border: none; padding: 14px;
        cursor: pointer; letter-spacing: 0.5px;
    }
    .stButton > button:hover { background: #5c6bc0; }
    .aviso { background: #1a1e2e; border-left: 3px solid #5c6bc0;
             padding: 12px 16px; border-radius: 4px; font-size: 0.85rem; color: #8892a4; }
    hr { border-color: #1e2130; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Carregamento do modelo ──────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    # Tenta caminhos relativos e dentro de /model
    caminhos_modelo = ["modelo_final.joblib", "model/modelo_final.joblib"]
    caminhos_scaler = ["scaler.joblib", "model/scaler.joblib"]
    modelo, scaler = None, None
    for p in caminhos_modelo:
        if os.path.exists(p):
            modelo = joblib.load(p)
            break
    for p in caminhos_scaler:
        if os.path.exists(p):
            scaler = joblib.load(p)
            break
    return modelo, scaler

modelo, scaler = carregar_modelo()

# ── Definição das features ──────────────────────────────────────────────────
FEATURES_WORD = [
    'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet',
    'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses',
    'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
    'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857',
    'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
    'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference',
]
FEATURES_CHAR = ['semicolon', 'parenthesis', 'bracket', 'exclamation', 'dollar', 'hash']
FEATURES_CAP  = ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="titulo">🛡️ Detector de Spam</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Grupo 19 — Luis, Davi e Erick &nbsp;|&nbsp; Fábrica de Projetos 3º Termo &nbsp;|&nbsp; UNIMAR 2026<br>Modelo SVM treinado no dataset <b>Spambase (UCI)</b> — classificação binária de e-mails</div>', unsafe_allow_html=True)

if modelo is None or scaler is None:
    st.error("⚠️ Modelo não encontrado. Certifique-se de que `modelo_final.joblib` e `scaler.joblib` estão na pasta do projeto.")
    st.stop()

st.markdown('<div class="aviso">💡 <b>Como usar:</b> Insira as frequências das palavras e caracteres presentes no e-mail que deseja analisar. Os valores representam a porcentagem de ocorrência em relação ao total de palavras/caracteres do e-mail.</div>', unsafe_allow_html=True)

# ── Formulário de entrada ───────────────────────────────────────────────────
with st.form("formulario_predicao"):

    # ── Frequência de palavras ──────────────────────────────────────────────
    st.markdown('<div class="secao">📝 Frequência de Palavras (% do total de palavras do e-mail)</div>', unsafe_allow_html=True)

    valores_word = {}
    cols_por_linha = 6
    chunks = [FEATURES_WORD[i:i+cols_por_linha] for i in range(0, len(FEATURES_WORD), cols_por_linha)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, feat in zip(cols, chunk):
            with col:
                valores_word[feat] = st.number_input(
                    f"word_freq_{feat}", min_value=0.0, max_value=100.0,
                    value=0.0, step=0.01, format="%.2f", key=f"word_{feat}"
                )

    # ── Frequência de caracteres ────────────────────────────────────────────
    st.markdown('<div class="secao">🔣 Frequência de Caracteres Especiais (% do total de caracteres)</div>', unsafe_allow_html=True)

    char_labels = {
        'semicolon':    'char_freq_ ;',
        'parenthesis':  'char_freq_ (',
        'bracket':      'char_freq_ [',
        'exclamation':  'char_freq_ !',
        'dollar':       'char_freq_ $',
        'hash':         'char_freq_ #',
    }
    valores_char = {}
    cols = st.columns(6)
    for col, feat in zip(cols, FEATURES_CHAR):
        with col:
            valores_char[feat] = st.number_input(
                char_labels[feat], min_value=0.0, max_value=100.0,
                value=0.0, step=0.001, format="%.3f", key=f"char_{feat}"
            )

    # ── Sequências de maiúsculas ────────────────────────────────────────────
    st.markdown('<div class="secao">🔠 Sequências de Letras Maiúsculas</div>', unsafe_allow_html=True)

    col1, col2, col3, _ = st.columns([1, 1, 1, 1])
    with col1:
        cap_avg = st.number_input("Comprimento médio das sequências", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    with col2:
        cap_long = st.number_input("Comprimento da sequência mais longa", min_value=0, value=1, step=1)
    with col3:
        cap_total = st.number_input("Total de letras maiúsculas no e-mail", min_value=0, value=1, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Analisar E-mail")

# ── Predição ────────────────────────────────────────────────────────────────
if submitted:
    entrada = (
        [valores_word[f] for f in FEATURES_WORD] +
        [valores_char[f] for f in FEATURES_CHAR] +
        [cap_avg, cap_long, cap_total]
    )
    X = np.array(entrada).reshape(1, -1)
    X_norm = scaler.transform(X)
    predicao = modelo.predict(X_norm)[0]
    probabilidade = modelo.predict_proba(X_norm)[0]
    prob_spam    = probabilidade[1] * 100
    prob_nospam  = probabilidade[0] * 100

    if predicao == 1:
        st.markdown(f"""
        <div class="resultado-spam">
            <div class="badge badge-spam">CLASSIFICAÇÃO DO MODELO</div>
            <div class="resultado-titulo" style="color:#e74c3c;">🚨 SPAM DETECTADO</div>
            <div class="resultado-prob">
                Probabilidade de SPAM: <b style="color:#e74c3c;">{prob_spam:.1f}%</b>
                &nbsp;|&nbsp;
                Probabilidade de Não-Spam: <b>{prob_nospam:.1f}%</b>
            </div>
            <p style="color:#c0392b; font-size:0.85rem; margin-top:14px;">
                ⚠️ Este e-mail apresenta características típicas de spam.<br>
                Recomenda-se não interagir com links ou anexos presentes nesta mensagem.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="resultado-ok">
            <div class="badge badge-ok">CLASSIFICAÇÃO DO MODELO</div>
            <div class="resultado-titulo" style="color:#2ecc71;">✅ E-MAIL LEGÍTIMO</div>
            <div class="resultado-prob">
                Probabilidade de Não-Spam: <b style="color:#2ecc71;">{prob_nospam:.1f}%</b>
                &nbsp;|&nbsp;
                Probabilidade de SPAM: <b>{prob_spam:.1f}%</b>
            </div>
            <p style="color:#27ae60; font-size:0.85rem; margin-top:14px;">
                ✔️ Este e-mail não apresenta padrões associados a spam.<br>
                O modelo classifica esta mensagem como legítima com base nas frequências informadas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Detalhes técnicos
    with st.expander("🔬 Ver detalhes técnicos da predição"):
        st.markdown(f"""
        **Modelo utilizado:** SVM (Support Vector Machine) com kernel RBF  
        **Pré-processamento:** StandardScaler (média 0, desvio padrão 1)  
        **Número de features:** 57  
        **Predição bruta:** `{predicao}` ({'spam' if predicao == 1 else 'não-spam'})  
        **Prob. não-spam:** `{prob_nospam:.4f}%`  
        **Prob. spam:** `{prob_spam:.4f}%`
        """)
        st.caption("Os valores informados foram normalizados pelo mesmo StandardScaler ajustado durante o treinamento, garantindo consistência com o modelo salvo.")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#3d4559; font-size:0.78rem; padding: 8px 0 24px 0;">
    Fábrica de Projetos — 3º Termo &nbsp;|&nbsp; UNIMAR 2026 &nbsp;|&nbsp;
    Dataset: Spambase UCI ML Repository &nbsp;|&nbsp; Modelo: SVM (scikit-learn)
</div>
""", unsafe_allow_html=True)
