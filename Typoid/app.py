# Updated Streamlit UI Code (Professional Gradient + Single Page Layout)

import streamlit as st
import numpy as np
import pickle

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Typhoid Prediction System",
    layout="wide"
)

# =============================
# LOAD MODEL
# =============================
model = pickle.load(
    open(r"F:\ML project\Typoid\Model\hybrid_model_new.pkl", "rb")
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb, #e8f5e9);
    background-attachment: fixed;
}

/* Remove extra top spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Main Container */
.main-box {
    background: rgba(255,255,255,0.92);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

/* Title */
.big-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #1565c0, #2e7d32);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #37474f;
    margin-bottom: 20px;
    font-weight: 500;
}

/* Section Titles */
.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #1b5e20;
    margin-bottom: 10px;
}

/* Labels */
label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #263238 !important;
}

/* Number Inputs */
input[type="number"] {
    font-size: 17px !important;
    height: 48px !important;
    border-radius: 10px !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    font-size: 17px !important;
    min-height: 48px !important;
    border-radius: 10px !important;
}

/* Button */
.stButton button {
    width: 100%;
    height: 55px;
    font-size: 22px !important;
    font-weight: bold;
    border-radius: 14px;
    border: none;
    color: white;
    background: linear-gradient(90deg, #1565c0, #2e7d32);
    transition: 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #0d47a1, #1b5e20);
}

/* Result Box */
.result-box {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    padding: 15px;
    border-radius: 15px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# =============================
# MAIN BOX START
# =============================
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown(
    '<div class="big-title">🧪 Typhoid Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter patient details and laboratory test values for prediction</div>',
    unsafe_allow_html=True
)

# =============================
# INPUT SECTION
# =============================
col_space1, col1, col2, col3, col_space2 = st.columns([0.2,1,1,1,0.2])

# COLUMN 1
with col1:

    st.markdown(
        '<div class="section-title">Patient Information</div>',
        unsafe_allow_html=True
    )

    Age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=25.0
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

# COLUMN 2
with col2:

    st.markdown(
        '<div class="section-title">Widal Test</div>',
        unsafe_allow_html=True
    )

    TO = st.number_input("TO", min_value=0.0)
    TH = st.number_input("TH", min_value=0.0)
    AH = st.number_input("AH", min_value=0.0)
    BH = st.number_input("BH", min_value=0.0)

# COLUMN 3
with col3:

    st.markdown(
        '<div class="section-title">Weil-Felix Test</div>',
        unsafe_allow_html=True
    )

    OX2 = st.number_input("OX2", min_value=0.0)
    OXK = st.number_input("OXK", min_value=0.0)
    OX9 = st.number_input("OX9", min_value=0.0)

    A = st.number_input("A", min_value=0.0)
    M = st.number_input("M", min_value=0.0)

# =============================
# ENCODING
# =============================
Gender = 1 if Gender == "Male" else 0

st.markdown("<div style='margin-top:5px;'></div>", unsafe_allow_html=True)

# =============================
# PREDICTION
# =============================
if st.button("🔍 Predict Typhoid"):

    input_data = np.array([[
        Age,
        Gender,
        TO,
        TH,
        AH,
        BH,
        OX2,
        OXK,
        OX9,
        A,
        M
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 0:

        st.warning("🟡 Minimal Typhoid Detected")

    elif prediction == 1:

        st.success("🟢 Negative Typhoid")

    else:

        st.error("🔴 Positive Typhoid Detected")

# =============================
# MAIN BOX END
# =============================
st.markdown('</div>', unsafe_allow_html=True)



