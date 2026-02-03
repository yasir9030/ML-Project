import streamlit as st
import numpy as np
import pickle

# PAGE CONFIG

st.set_page_config(
    page_title="Typhoid Prediction System",
    layout="wide"
)

# LOAD MODEL

model = pickle.load(open("Model/Typhoid  prediction.sav", "rb"))


# CUSTOM CSS (BIG INPUTS + TEXT)

st.markdown("""
<style>
/* Title */
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
}

/* Section headings */
.section-title {
    font-size: 26px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #2e7d32;
}

/* Labels */
label {
    font-size: 20px !important;
    font-weight: bold !important;
}

/* Number input & select box text */
input, select {
    font-size: 20px !important;
    height: 45px !important;
}

/* Dropdown selected text */
div[data-baseweb="select"] > div {
    font-size: 20px !important;
}

/* Button */
.stButton button {
    font-size: 22px !important;
    font-weight: bold;
    height: 55px;
}
</style>
""", unsafe_allow_html=True)


# TITLE
st.markdown('<div class="big-title">🧪 Typhoid Prediction System</div>', unsafe_allow_html=True)
st.write("Enter patient details and laboratory test values")

st.markdown("---")

# =============================
# INPUT SECTION (4 COLUMNS)
# =============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="section-title">Patient Info</div>', unsafe_allow_html=True)
    Age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=28.0)
    Gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    st.markdown('<div class="section-title">Widal Test</div>', unsafe_allow_html=True)
    TO = st.number_input("TO", 0.0)
    TH = st.number_input("TH", 0.0)
    AH = st.number_input("AH", 0.0)
    BH = st.number_input("BH", 0.0)

with col3:
    st.markdown('<div class="section-title">Weil-Felix</div>', unsafe_allow_html=True)
    OX2 = st.number_input("OX2", 0.0)
    OXK = st.number_input("OXK", 0.0)
    OX9 = st.number_input("OX9", 0.0)

with col4:
    st.markdown('<div class="section-title">Other Info</div>', unsafe_allow_html=True)
    A = st.number_input("A", 0.0)
    M = st.number_input("M", 0.0)
    Rickettsia = st.selectbox("Rickettsia Suspect", ["No", "Yes"])
    Acute = st.selectbox("Acute Typhoid", ["No", "Yes"])
    ParaA = st.selectbox("Paratyphoid A", ["No", "Yes"])
    ParaB = st.selectbox("Paratyphoid B", ["No", "Yes"])

# =============================
# ENCODING
# =============================
Gender = 1 if Gender == "Male" else 0
Rickettsia = 1 if Rickettsia == "Yes" else 0
Acute = 1 if Acute == "Yes" else 0
ParaA = 1 if ParaA == "Yes" else 0
ParaB = 1 if ParaB == "Yes" else 0

Name_encoded = 0  # dummy value

# =============================
# PREDICTION
# =============================
st.markdown("---")

if st.button("🔍 Predict Typhoid", use_container_width=True):
    input_data = np.array([[ 
        Name_encoded,
        Age,
        Gender,
        TO, TH, AH, BH,
        OX2, OXK, OX9,
        A, M,
        Rickettsia,
        Acute,
        ParaA,
        ParaB
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)

    if prediction == 0:
        st.warning("🟡 **Minimal Typhoid**")
    elif prediction == 1:
        st.success("🟢 **Negative Typhoid**")
    else:
        st.error("🔴 **Positive Typhoid**")
