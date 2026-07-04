import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Clasificación de Frutas IA_ISC",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .main-title {
            text-align: center;
            color: #E65100;
            font-weight: 700;
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
        }
        
        .subtitle {
            text-align: center;
            color: #EF6C00;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .result-card {
            background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
            border-radius: 20px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 8px 32px rgba(255, 111, 0, 0.2);
            border: 1px solid #FFCC80;
        }
        
        .winner-badge {
            display: inline-block;
            background: linear-gradient(135deg, #E65100, #FB8C00);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.3rem;
            box-shadow: 0 4px 15px rgba(230, 81, 0, 0.3);
            margin-bottom: 1rem;
        }
        
        .footer {
            text-align: center;
            color: #9E9E9E;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #EEEEEE;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# ENCABEZADO
# ============================================================
st.markdown("🍎")
st.markdown('<h1 class="main-title">Clasificador de Frutas con IA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">IA-ISC • Fruits-360 • 2026 • Angeles Euceda</p>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #616161; margin-bottom: 2rem;">
        🍇 Sube una imagen y la IA identificará la fruta automáticamente
    </div>
""", unsafe_allow_html=True)

# ============================================================
# MODELO
# ============================================================
IMG_SIZE = (224, 224)
MODEL_DIR = Path("Clasificacion_flores_mobilenet")  # puedes cambiar carpeta si quieres
CLASS_PATH = MODEL_DIR / "class_names.json"
MODEL_PATHS = [
    MODEL_DIR / "Clasificacion_flores_mobilenet.keras",
    MODEL_DIR / "flores_mobilenet.h5"
]

# ============================================================
# FUNCIONES INTELIGENTES
# ============================================================

FRUIT_EMOJIS = {
    "apple": "🍎",
    "banana": "🍌",
    "orange": "🍊",
    "grape": "🍇",
    "strawberry": "🍓",
    "watermelon": "🍉",
    "lemon": "🍋",
    "peach": "🍑",
    "pear": "🍐",
    "pineapple": "🍍",
    "coconut": "🥥",
    "kiwi": "🥝",
    "tomato": "🍅",
    "carrot": "🥕",
    "pepper": "🌶️",
    "cucumber": "🥒",
    "eggplant": "🍆",
    "avocado": "🥑",
    "melon": "🍈",
    "mango": "🥭"
}

def limpiar_nombre(clase):
    return " ".join(clase.lower().split()[:-1])

def obtener_emojis(clase):
    base = limpiar_nombre(clase)
    for k, v in FRUIT_EMOJIS.items():
        if k in base:
            return v
    return "🍏"

def formato_nombre(clase):
    return limpiar_nombre(clase).title()

# ============================================================
# CARGA MODELO
# ============================================================
@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("❌ No se encontró el modelo.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    top3 = np.argsort(preds)[-3:][::-1]
    return [
        (clases[i], float(preds[i]) * 100)
        for i in top3
    ]

# ============================================================
# INICIALIZAR
# ============================================================
with st.spinner("🔄 Cargando modelo..."):
    modelo = cargar_modelo()
    clases = cargar_clases()

# ============================================================
# UPLOADER
# ============================================================
archivo = st.file_uploader(
    "📷 Sube una imagen de fruta",
    type=["jpg", "jpeg", "png"]
)

if archivo:

    imagen = Image.open(archivo)

    st.image(imagen, caption="🖼️ Imagen subida", use_container_width=True)

    with st.spinner("🔍 Analizando..."):
        resultados = predecir(imagen)

    ganador_key, ganador_prob = resultados[0]

    icono = obtener_emojis(ganador_key)
    nombre = formato_nombre(ganador_key)

    # ========================================================
    # RESULTADO
    # ========================================================
    st.markdown(f"""
        <div class="result-card">
            <div style="text-align:center;">
                <div style="font-size:4rem;">{icono}</div>
                <div class="winner-badge">{nombre}</div>
                <p style="font-size:1.2rem; margin-top:1rem;">
                    Confianza: <b>{ganador_prob:.2f}%</b>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align:center; padding:3rem; color:#999;">
            <div style="font-size:4rem;">🍎</div>
            <p>Sube una imagen de fruta para comenzar</p>
        </div>
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        🍎 Clasificador de Frutas IA • MobileNetV2 • 2026
    </div>
""", unsafe_allow_html=True)
