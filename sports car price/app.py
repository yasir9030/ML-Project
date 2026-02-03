import pickle
import numpy as np
import streamlit as st
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
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
st.markdown("<h1 style='text-align:center;'>🚗 Car Price Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================= LOAD MODEL =================
model_path = r"F:\ML project\car_price\savedmodel\car_price_model.sav"

if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()

car_model = pickle.load(open(model_path, "rb"))

# ================= LAYOUT =================
col1, col2 = st.columns(2)

# ================= LEFT COLUMN =================
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚘 Car Information")

    car_make = st.selectbox("Car Make", ["Toyota", "Honda", "BMW", "Audi", "Ford"])
    car_model_name = st.text_input("Car Model")
    year = st.number_input("Manufacturing Year", 1990, 2025, 2020)

    engine_size = st.number_input("Engine Size (L)", 0.8, 8.0, 2.0)
    horsepower = st.number_input("Horsepower", 50, 1500, 150)
    torque = st.number_input("Torque (lb-ft)", 50, 1500, 200)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RIGHT COLUMN =================
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Performance")

    zero_to_sixty = st.number_input("0–60 MPH Time (seconds)", 2.0, 20.0, 8.0)

    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])

    st.markdown("</div>", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================
def transmission_map(val):
    return 0 if val == "Manual" else 1

def fuel_map(val):
    return {"Petrol": 0, "Diesel": 1, "Electric": 2, "Hybrid": 3}[val]

def make_map(val):
    return {"Toyota": 0, "Honda": 1, "BMW": 2, "Audi": 3, "Ford": 4}[val]

# ================= PREDICTION =================
st.markdown("---")

if st.button("💰 Predict Car Price", use_container_width=True):
    try:
        final_input = np.array([[ 
            make_map(car_make),
            year,
            engine_size,
            horsepower,
            torque,
            zero_to_sixty,
            transmission_map(transmission),
            fuel_map(fuel_type)
        ]])

        # Handle feature mismatch
        expected_features = car_model.n_features_in_
        current_features = final_input.shape[1]

        if current_features < expected_features:
            padding = expected_features - current_features
            final_input = np.hstack((final_input, np.zeros((1, padding))))

        prediction = car_model.predict(final_input)

        st.success(f"💵 Estimated Car Price: **${prediction[0]:,.2f} USD**")

    except Exception as e:
        st.error("❌ Prediction Failed")
        st.exception(e)
