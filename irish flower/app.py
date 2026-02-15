import streamlit as st
import numpy as np
import pickle

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Iris Flower Classification System",
    layout="wide"
)

# =============================
# LOAD MODEL
# =============================
model = pickle.load(open("MODEL/iris flower classification  prediction.sav", "rb"))

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
}

.section-title {
    font-size: 26px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #2e7d32;
}

label {
    font-size: 20px !important;
    font-weight: bold !important;
}

input {
    font-size: 20px !important;
    height: 45px !important;
}

.stButton button {
    font-size: 22px !important;
    font-weight: bold;
    height: 55px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown('<div class="big-title">🌸 Iris Flower Classification System</div>', unsafe_allow_html=True)
st.write("Enter flower measurements to predict the species")

st.markdown("---")

# =============================
# INPUT SECTION (4 COLUMNS)
# =============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="section-title">Sepal Length</div>', unsafe_allow_html=True)
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)

with col2:
    st.markdown('<div class="section-title">Sepal Width</div>', unsafe_allow_html=True)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)

with col3:
    st.markdown('<div class="section-title">Petal Length</div>', unsafe_allow_html=True)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)

with col4:
    st.markdown('<div class="section-title">Petal Width</div>', unsafe_allow_html=True)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

# =============================
# PREDICTION
# =============================
st.markdown("---")

if st.button("🔍 Predict Flower Species", use_container_width=True):
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)

    if prediction == 0:
        st.success("🌼 **Iris Setosa**")
    elif prediction == 1:
        st.success("🌸 **Iris Versicolor**")
    else:
        st.success("🌺 **Iris Virginica**")
