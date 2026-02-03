import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Dengue Prediction using Machine Learning",
    layout="centered",
    page_icon="🦟"
)

st.title("🦟 Dengue Prediction System")

# ================= LOAD MODEL =================
working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(working_dir, "save model", "Dengue_Prediction_Model.sav")
encoder_path = os.path.join(working_dir, "save model", "encoder.sav")

dengue_model = pickle.load(open(model_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))

# ================= USER INPUT =================
st.subheader("Enter Patient Details")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)

ns1 = st.selectbox("NS1 Test", ["Positive", "Negative"])
igg = st.selectbox("IgG Test", ["Positive", "Negative"])
igm = st.selectbox("IgM Test", ["Positive", "Negative"])

area_list = [
    'Mirpur','Chawkbazar','Paltan','Motijheel','Gendaria','Dhanmondi',
    'New Market','Sher-e-Bangla Nagar','Kafrul','Pallabi','Mohammadpur',
    'Shahbagh','Shyampur','Kalabagan','Bosila','Jatrabari','Adabor',
    'Kamrangirchar','Biman Bandar','Ramna','Badda','Bangshal','Sabujbagh',
    'Hazaribagh','Sutrapur','Lalbagh','Demra','Banasree','Cantonment',
    'Keraniganj','Tejgaon','Khilkhet','Kadamtali','Gulshan','Rampura','Khilgaon'
]

area = st.selectbox("Area", sorted(area_list))
area_type = st.selectbox("Area Type", ["Undeveloped", "Developed"])
house_type = st.selectbox("House Type", ["Building", "Other", "Tinshed"])
district = st.selectbox("District", ["Dhaka"])

# ================= HELPER =================
def binary_map(val):
    return 1 if val == "Positive" else 0

# ================= PREDICTION =================
if st.button("Predict Dengue"):
    try:
        # ---- Numeric Features ----
        numeric_features = np.array([[
            age,
            binary_map(ns1),
            binary_map(igg),
            binary_map(igm)
        ]])

        # ---- Categorical Features ----
        cat_df = pd.DataFrame(
            [[gender, area, area_type, house_type, district]],
            columns=["Gender", "Area", "AreaType", "HouseType", "District"]
        )

        encoded_cat = encoder.transform(cat_df)
        if hasattr(encoded_cat, "toarray"):
            encoded_cat = encoded_cat.toarray()

        # ---- Combine Features ----
        final_input = np.hstack((numeric_features, encoded_cat))

        # ---- AUTO FEATURE MATCH ----
        expected_features = dengue_model.n_features_in_

        if final_input.shape[1] > expected_features:
            final_input = final_input[:, :expected_features]
        elif final_input.shape[1] < expected_features:
            diff = expected_features - final_input.shape[1]
            final_input = np.hstack(
                (final_input, np.zeros((1, diff)))
            )

        # ---- MEDICAL RULE OVERRIDE ----
        ns1_val = binary_map(ns1)
        igg_val = binary_map(igg)
        igm_val = binary_map(igm)
        
        if ns1_val == 0 and igg_val == 0 and igm_val == 0:
            st.success("✅ No Dengue Detected (Medical Rule)")
        elif ns1_val == 1 and igg_val == 1 and igm_val == 0:
            st.error("⚠️ Dengue Detected (Medical Rule)") 
        elif ns1_val==1 and igg_val==0 and igm_val==0:
            st.error(" Dengue Detected ")
        elif ns1_val == 1 and igg_val == 1 and igm_val == 1:
            st.error("⚠️ Dengue Detected (Medical Rule)") 
        elif ns1_val==1 and igg_val==0 and igm_val==1:
            st.error("No Dengue Detected ")
       
        else:
            prediction = dengue_model.predict(final_input)

            if prediction[0] == 1:
                st.error("⚠️ Dengue Detected")
            else:
                st.success("✅ No Dengue Detected")

    except Exception as e:
        st.error("❌ Prediction Failed")
        st.exception(e)