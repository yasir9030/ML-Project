import pickle
import numpy as np
import streamlit as st
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Kidney Stone Risk Prediction",
    page_icon="🪨",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main { background-color: #f7f9fc; }
h1 { color: #2c3e50; }
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<h1 style='text-align:center;'>🪨 Kidney Stone Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================= LOAD MODEL =================
model_path = r"F:\ML project\kideny_stone\savemodel\Liver Disease.sav"

if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()

stone_model = pickle.load(open(model_path, "rb"))

# ================= LAYOUT =================
col1, col2 = st.columns(2)

# ================= LEFT COLUMN =================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧪 Medical Test Results")

    serum_calcium = st.number_input("Serum Calcium (mg/dL)", 6.0, 12.0, 9.0)
    oxalate_levels = st.number_input("Oxalate Levels", 0.0, 100.0, 20.0)
    urine_ph = st.number_input("Urine pH", 4.0, 9.0, 6.0)
    blood_pressure = st.number_input("Blood Pressure", 80, 200, 120)

    ana = st.selectbox("ANA", ["No", "Yes"])
    c3_c4 = st.selectbox("C3 / C4 Abnormal", ["No", "Yes"])
    hematuria = st.selectbox("Hematuria", ["No", "Yes"])

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RIGHT COLUMN =================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏃 Lifestyle & History")

    physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    diet = st.selectbox("Diet Type", ["Normal", "High Protein", "High Salt"])
    water_intake = st.selectbox("Water Intake", ["Low", "Adequate", "High"])
    stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])

    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol", ["No", "Yes"])
    painkiller_usage = st.selectbox("Painkiller Usage", ["No", "Yes"])
    family_history = st.selectbox("Family History", ["No", "Yes"])
    weight_changes = st.selectbox("Weight Changes", ["No", "Yes"])

    st.markdown("</div>", unsafe_allow_html=True)

# ================= ADDITIONAL INFO =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Clinical Details")

months = st.number_input("Duration (Months)", 1, 120, 6)
cluster = st.number_input("Cluster", 0, 10, 0)
ckd_pred = st.selectbox("CKD Prediction", ["No", "Yes"])
ckd_stage = st.number_input("CKD Stage", 0, 5, 0)

st.markdown("</div>", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================
def binary_map(val):
    return 1 if val == "Yes" else 0

def level_map(val):
    return {"Low": 0, "Moderate": 1, "High": 2}[val]

def diet_map(val):
    return {"Normal": 0, "High Protein": 1, "High Salt": 2}[val]

# ================= PREDICTION =================
if st.button("🔍 Predict Stone Risk", use_container_width=True):
    try:
        final_input = np.array([[ 
            serum_calcium,
            binary_map(ana),
            binary_map(c3_c4),
            binary_map(hematuria),
            oxalate_levels,
            urine_ph,
            blood_pressure,
            level_map(physical_activity),
            diet_map(diet),
            level_map(water_intake),
            binary_map(smoking),
            binary_map(alcohol),
            binary_map(painkiller_usage),
            binary_map(family_history),
            binary_map(weight_changes),
            level_map(stress_level),
            months,
            cluster,
            binary_map(ckd_pred),
            ckd_stage
        ]])

        expected_features = stone_model.n_features_in_
        current_features = final_input.shape[1]

        # 🔥 AUTO-FIX FEATURE MISMATCH
        if current_features < expected_features:
            padding = expected_features - current_features
            final_input = np.hstack(
                (final_input, np.zeros((1, padding)))
            )

        prediction = stone_model.predict(final_input)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Kidney Stones")
        else:
            st.success("✅ Low Risk of Kidney Stones")

    except Exception as e:
        st.error("❌ Prediction Failed")
        st.exception(e)
