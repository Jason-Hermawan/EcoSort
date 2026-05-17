import os
import cv2
import numpy as np
import joblib
import streamlit as st
from PIL import Image
from skimage.feature import local_binary_pattern, hog

IMG_SIZE = 128
try:
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
except NameError:
    MODEL_DIR = "saved_model"

SUBCATEGORY_LABELS = {
    "plastic":   "Plastik",
    "metal":     "Logam",
    "glass":     "Kaca",
    "paper":     "Kertas",
    "cardboard": "Kardus",
    "battery":   "Baterai (B3)",
}

SUBCATEGORY_ICONS = {
    "plastic":   "🥤",
    "metal":     "🥫",
    "glass":     "🍾",
    "paper":     "📄",
    "cardboard": "📦",
    "battery":   "🔋",
}

SUBCATEGORY_TIPS = {
    "plastic":   "Cuci bersih sebelum didaur ulang. Pisahkan tutup botol dan label.",
    "metal":     "Bilas residu makanan. Logam dapat didaur ulang berulang kali.",
    "glass":     "Hati-hati pecahan kaca. Pisahkan berdasarkan warna jika memungkinkan.",
    "paper":     "Pastikan kering dan tidak terkontaminasi minyak/makanan.",
    "cardboard": "Lipat untuk menghemat tempat. Lepas tape dan staples jika ada.",
    "battery":   "Limbah B3! Jangan dibuang sembarangan. Serahkan ke drop point khusus baterai.",
}

st.set_page_config(
    page_title="EcoSort - Klasifikasi Sampah",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    "<style>"
    ".stApp { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%) !important; background-attachment: fixed !important; }"
    ".main .block-container { background: #ffffff !important; border-radius: 24px; padding: 2.5rem 2.5rem 2rem; margin-top: 1.5rem; max-width: 720px; box-shadow: 0 20px 60px rgba(46, 125, 50, 0.15); border: 1px solid rgba(76, 175, 80, 0.1); }"
    "#MainMenu, footer, header { visibility: hidden; }"
    ".stDeployButton { display: none; }"
    "[data-testid='collapsedControl'] { display: none; }"
    ".title-wrapper { text-align: center; margin-bottom: 1.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #e8f5e9; }"
    ".logo-text { font-size: 2.8rem; font-weight: 800; color: #1b5e20; margin: 0; letter-spacing: -1px; line-height: 1.1; }"
    ".logo-text .accent { background: linear-gradient(135deg, #2e7d32, #66bb6a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }"
    ".tagline { color: #546e7a; font-size: 0.95rem; font-weight: 400; margin: 0.4rem 0 0.6rem; }"
    ".uni-pill { display: inline-block; background: #e8f5e9; color: #1b5e20; font-weight: 500; font-size: 0.78rem; padding: 5px 14px; border-radius: 100px; letter-spacing: 0.3px; }"
    ".section-label { font-size: 0.78rem; font-weight: 600; color: #66bb6a; text-transform: uppercase; letter-spacing: 1.5px; margin: 1.8rem 0 0.8rem; }"

    "[data-testid='stFileUploader'] { background: transparent !important; }"
    "[data-testid='stFileUploaderDropzone'] { background: #f1f8e9 !important; border: 2px dashed #66bb6a !important; border-radius: 16px !important; padding: 1.5rem !important; min-height: 100px !important; }"
    "[data-testid='stFileUploaderDropzone']:hover { border-color: #2e7d32 !important; background: #e8f5e9 !important; }"

    "[data-testid='stFileUploaderDropzone'] p, [data-testid='stFileUploaderDropzone'] span { color: #2e7d32 !important; font-weight: 500 !important; }"
    "[data-testid='stFileUploaderDropzone'] small { color: #66bb6a !important; font-size: 0.8rem !important; }"

    "[data-testid='stFileUploaderDropzone'] button { background: #2e7d32 !important; color: white !important; border-radius: 10px !important; font-weight: 600 !important; border: none !important; padding: 0.5rem 1.2rem !important; }"
    "[data-testid='stFileUploaderDropzone'] button:hover { background: #1b5e20 !important; }"

    "[data-testid='stFileUploaderDropzone'] button p, [data-testid='stFileUploaderDropzone'] button span, [data-testid='stFileUploaderDropzone'] button svg, [data-testid='stFileUploaderDropzone'] button path { color: #ffffff !important; fill: #ffffff !important; margin: 0 !important; }"

    "section[data-testid='stFileUploaderDropzone'] { background: #f1f8e9 !important; }"

    ".uploadedFile { background: white !important; border-radius: 8px; }"
    ".stImage > img { border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }"
    ".result-card-organik { background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%); border-radius: 22px; padding: 2rem 1.8rem 1.8rem; color: white; text-align: center; box-shadow: 0 12px 32px rgba(46, 125, 50, 0.3); margin-top: 0.5rem; }"
    ".result-card-anorganik { background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); border-radius: 22px; padding: 2rem 1.8rem 1.8rem; color: white; text-align: center; box-shadow: 0 12px 32px rgba(21, 101, 192, 0.3); margin-top: 0.5rem; }"
    ".result-card-organik .icon, .result-card-anorganik .icon { font-size: 3.2rem; line-height: 1; margin-bottom: 0.4rem; }"
    ".result-card-organik .main-label, .result-card-anorganik .main-label { font-size: 2rem; font-weight: 800; letter-spacing: 2px; margin: 0.3rem 0; text-transform: uppercase; color: white !important; }"
    ".result-card-organik .confidence, .result-card-anorganik .confidence { display: inline-block; background: rgba(255,255,255,0.22); padding: 6px 18px; border-radius: 100px; font-size: 0.92rem; font-weight: 500; margin-top: 0.8rem; color: white !important; }"
    ".bar-row { margin: 0.8rem 0; }"
    ".bar-label { display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 500; color: #37474f; margin-bottom: 0.4rem; }"
    ".bar-label .pct { font-weight: 600; color: #263238; }"
    ".bar-track { background: #f1f8e9; border-radius: 100px; overflow: hidden; height: 10px; }"
    ".bar-fill-organik { height: 100%; background: linear-gradient(90deg, #2e7d32, #66bb6a); border-radius: 100px; }"
    ".bar-fill-anorganik { height: 100%; background: linear-gradient(90deg, #1565c0, #42a5f5); border-radius: 100px; }"
    ".tip-box { background: #fff8e1; border-left: 4px solid #ffb300; padding: 1rem 1.2rem; border-radius: 12px; margin-top: 1.2rem; color: #4e342e; font-size: 0.92rem; line-height: 1.55; }"
    ".tip-box .tip-title { font-weight: 600; color: #e65100; display: block; margin-bottom: 0.3rem; font-size: 0.95rem; }"
    ".subcat-detail { background: rgba(255,255,255,0.18); border-radius: 14px; padding: 0.8rem 1.2rem; margin-top: 0.8rem; color: white !important; font-size: 0.95rem; font-weight: 500; }"
    ".subcat-detail .subcat-icon { font-size: 1.4rem; margin-right: 0.4rem; vertical-align: middle; }"
    ".empty-state { text-align: center; padding: 3rem 1.5rem; color: #90a4ae; }"
    ".empty-state .icon-big { font-size: 4rem; opacity: 0.4; margin-bottom: 0.6rem; }"
    ".footer { text-align: center; font-size: 0.78rem; color: #78909c; margin-top: 2rem; padding-top: 1.2rem; border-top: 1px solid #eceff1; line-height: 1.6; }"
    ".stSpinner > div { border-top-color: #2e7d32 !important; }"
    "</style>",
    unsafe_allow_html=True
)


@st.cache_resource
def load_models():
    main = {
        "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "pca":    joblib.load(os.path.join(MODEL_DIR, "pca.pkl")),
        "svm":    joblib.load(os.path.join(MODEL_DIR, "svm_rbf_best.pkl")),
    }
    sub_path = os.path.join(MODEL_DIR, "svm_subcat.pkl")
    sub = None
    if os.path.exists(sub_path):
        sub = {
            "scaler":  joblib.load(os.path.join(MODEL_DIR, "scaler_subcat.pkl")),
            "pca":     joblib.load(os.path.join(MODEL_DIR, "pca_subcat.pkl")),
            "svm":     joblib.load(sub_path),
            "classes": joblib.load(os.path.join(MODEL_DIR, "subcat_classes.pkl")),
        }
    return main, sub


def preprocess(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = img.astype(np.float32) / 255.0
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 0] /= 180.0
    img_hsv[:, :, 1] /= 255.0
    img_hsv[:, :, 2] /= 255.0
    return img_rgb, img_hsv


def extract_all_features(img_rgb, img_hsv):
    img_hsv_uint8 = (img_hsv * np.array([180, 255, 255])).astype(np.uint8)
    h = cv2.calcHist([img_hsv_uint8], [0], None, [32], [0, 180]).flatten()
    s = cv2.calcHist([img_hsv_uint8], [1], None, [32], [0, 256]).flatten()
    v = cv2.calcHist([img_hsv_uint8], [2], None, [32], [0, 256]).flatten()
    h /= (h.sum() + 1e-7)
    s /= (s.sum() + 1e-7)
    v /= (v.sum() + 1e-7)
    color_hist = np.concatenate([h, s, v])

    gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    if len(lbp_hist) < 256:
        lbp_hist = np.pad(lbp_hist, (0, 256 - len(lbp_hist)))
    else:
        lbp_hist = lbp_hist[:256]

    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")

    return np.concatenate([color_hist, lbp_hist, hog_feat])


# HEADER
st.markdown(
    "<div class='title-wrapper'>"
    "<h1 class='logo-text'>♻️ <span class='accent'>EcoSort</h1>"
    "<p class='tagline'>Klasifikasi Sampah Organik & Anorganik</p>"
    "<span class='uni-pill'>Kelompok 8 · Kelas LG01</span>"
    "</div>",
    unsafe_allow_html=True
)


try:
    main_model, sub_model = load_models()
except Exception as e:
    st.error("Model belum tersimpan. Detail: " + str(e))
    st.stop()


# UPLOAD SECTION
st.markdown("<div class='section-label'>Unggah Gambar Sampah</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih gambar (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# RESULT SECTION
if uploaded_file is None:
    st.markdown(
        "<div class='empty-state'>"
        "<div class='icon-big'>📷</div>"
        "<p>Unggah gambar sampah untuk memulai klasifikasi</p>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    img_pil = Image.open(uploaded_file)
    st.image(img_pil, use_container_width=True)

    with st.spinner("Menganalisis gambar..."):
        img_rgb, img_hsv = preprocess(img_pil)
        feat = extract_all_features(img_rgb, img_hsv).reshape(1, -1)
        feat_scaled = main_model["scaler"].transform(feat)
        feat_pca = main_model["pca"].transform(feat_scaled)
        pred = main_model["svm"].predict(feat_pca)[0]
        proba = main_model["svm"].predict_proba(feat_pca)[0]

        subcat_name = None
        subcat_key = None
        subcat_conf = None
        if pred == 1 and sub_model is not None:
            feat_scaled_sub = sub_model["scaler"].transform(feat)
            feat_pca_sub = sub_model["pca"].transform(feat_scaled_sub)
            subcat_pred = sub_model["svm"].predict(feat_pca_sub)[0]
            subcat_proba_arr = sub_model["svm"].predict_proba(feat_pca_sub)[0]
            subcat_key = subcat_pred
            subcat_name = SUBCATEGORY_LABELS.get(subcat_pred, subcat_pred.title())
            idx = list(sub_model["classes"]).index(subcat_pred)
            subcat_conf = subcat_proba_arr[idx]

    st.markdown("<div class='section-label'>Hasil Klasifikasi</div>", unsafe_allow_html=True)

    conf_pct = proba[pred] * 100

    if pred == 0:
        st.markdown(
            "<div class='result-card-organik'>"
            "<div class='icon'>🌿</div>"
            "<div class='main-label'>Organik</div>"
            "<div class='confidence'>Confidence: " + "{:.1f}".format(conf_pct) + "%</div>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        subcat_html = ""
        if subcat_name:
            subcat_icon = SUBCATEGORY_ICONS.get(subcat_key, "🔩")
            subcat_html = (
                "<div class='subcat-detail'>"
                "<span class='subcat-icon'>" + subcat_icon + "</span>"
                "Jenis: <strong>" + subcat_name + "</strong>"
                "&nbsp;·&nbsp; " + "{:.1f}".format(subcat_conf * 100) + "%"
                "</div>"
            )
        st.markdown(
            "<div class='result-card-anorganik'>"
            "<div class='icon'>🔩</div>"
            "<div class='main-label'>Anorganik</div>"
            "<div class='confidence'>Confidence: " + "{:.1f}".format(conf_pct) + "%</div>"
            + subcat_html +
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='section-label'>Distribusi Confidence</div>", unsafe_allow_html=True)

    org_pct = proba[0] * 100
    inorg_pct = proba[1] * 100

    st.markdown(
        "<div class='bar-row'>"
        "<div class='bar-label'><span>🌿 Organik</span><span class='pct'>" + "{:.1f}".format(org_pct) + "%</span></div>"
        "<div class='bar-track'><div class='bar-fill-organik' style='width: " + str(org_pct) + "%;'></div></div>"
        "</div>"
        "<div class='bar-row'>"
        "<div class='bar-label'><span>🔩 Anorganik</span><span class='pct'>" + "{:.1f}".format(inorg_pct) + "%</span></div>"
        "<div class='bar-track'><div class='bar-fill-anorganik' style='width: " + str(inorg_pct) + "%;'></div></div>"
        "</div>",
        unsafe_allow_html=True
    )

    if pred == 0:
        tip_title = "Pengelolaan Sampah Organik"
        tip_text = "Sampah organik dapat diolah menjadi <strong>kompos</strong> atau <strong>pupuk cair</strong>. Pisahkan dari sampah anorganik dan hindari kontaminasi plastik."
    else:
        if subcat_key and subcat_key in SUBCATEGORY_TIPS:
            tip_title = "Tips untuk " + SUBCATEGORY_LABELS[subcat_key]
            tip_text = SUBCATEGORY_TIPS[subcat_key]
        else:
            tip_title = "Pengelolaan Sampah Anorganik"
            tip_text = "Pilah berdasarkan jenis (plastik, kaca, logam, kertas, B3). Cuci bersih sebelum diserahkan ke bank sampah atau didaur ulang."

    st.markdown(
        "<div class='tip-box'>"
        "<span class='tip-title'>💡 " + tip_title + "</span>"
        + tip_text +
        "</div>",
        unsafe_allow_html=True
    )


st.markdown(
    "<div class='footer'>"
    "Klasifikasi Sampah Organik dan Anorganik Menggunakan Machine Learning Berbasis Fitur Visual<br>"
    "Computer Vision Project · School of Computer Science · BINUS University"
    "</div>",
    unsafe_allow_html=True
)
