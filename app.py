import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# ---------------- LOAD MODEL SAFELY ----------------
model_path = os.path.join(os.getcwd(), 'xgboost_bleach_model.pkl')

try:
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    # Safe extraction
    if isinstance(bundle, dict):
        model = bundle.get('model', bundle)
        scaler = bundle.get('scaler', None)
        label_encoder = bundle.get('label_encoder', None)
    else:
        model = bundle
        scaler = None
        label_encoder = None

    st.success("✅ Model Loaded Successfully")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ---------------- FEATURE LIST ----------------
feature_cols = [
    'Temperature_Celsius', 'SSTA', 'DHW_Stress', 'Hotspot',
    'ONI_Index', 'Chlorophyll_A_mgm3', 'Salinity', 'Dissolved_O2',
    'pH', 'Nitrate', 'Depth_m', 'Distance_to_Shore_km',
    'NDVI', 'NDWI', 'Turbidity_NTU', 'Month_sin', 'Month_cos',
    'Quarter', 'Temp_lag1', 'Temp_lag4', 'DHW_lag1', 'DHW_lag4',
    'Bleach_lag1', 'Bleach_lag4', 'Temp_rolling4w',
    'DHW_rolling4w', 'Bleach_rolling8w',
    'Thermal_Anomaly_Index', 'Oxygen_pH_Stress',
    'Env_Stress_Composite'
]

# ---------------- LOCATION DATA ----------------
location_species_data = {
    "Rameswaram": ["Acropora", "Porites", "Montipora"],
    "Tuticorin": ["Favia", "Goniastrea", "Acropora"],
    "Gulf of Mannar": ["Porites", "Favites", "Montipora"],
    "Andaman": ["Acropora", "Pocillopora", "Fungia"]
}

# NEW: donor sites
donor_sites = {
    "Acropora": "Rameswaram Reef",
    "Porites": "Mandapam Reef",
    "Montipora": "Pamban Reef",
    "Favia": "Tuticorin Reef",
    "Goniastrea": "Vembar Reef",
    "Favites": "Keelakarai Reef",
    "Pocillopora": "Andaman Reef",
    "Fungia": "Havelock Reef"
}

# ---------------- TITLE ----------------
st.title("🌊 Coral Species Recommendation System")
st.write("Recommend best coral species based on location & environment")

# ---------------- LOCATION ----------------
location = st.selectbox("📍 Select Location", list(location_species_data.keys()))

st.write(f"### 🌿 Previously grown species in {location}:")
st.write(", ".join(location_species_data[location]))

# ---------------- IMPORTANT FEATURES ----------------
st.subheader("⚠️ Key Environmental Inputs")

important_features = [
    "Temperature_Celsius",
    "DHW_Stress",
    "SSTA",
    "Hotspot",
    "pH",
    "Dissolved_O2",
    "Salinity",
    "Turbidity_NTU",
    "Depth_m",
    "Env_Stress_Composite"
]

default_values = {
    "Temperature_Celsius": 28.0,
    "DHW_Stress": 2.0,
    "SSTA": 1.0,
    "Hotspot": 0.5,
    "pH": 8.1,
    "Dissolved_O2": 6.0,
    "Salinity": 35.0,
    "Turbidity_NTU": 1.0,
    "Depth_m": 10.0,
    "Env_Stress_Composite": 0.5
}

user_inputs = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(important_features):
    if i % 2 == 0:
        user_inputs[feature] = col1.number_input(
            feature, value=default_values.get(feature, 0.0)
        )
    else:
        user_inputs[feature] = col2.number_input(
            feature, value=default_values.get(feature, 0.0)
        )

# ---------------- OPTIONAL FEATURES ----------------
st.markdown("---")
st.subheader("⚙️ Advanced Inputs (Optional)")

optional_inputs = {}

with st.expander("🔽 Show All Features"):
    for col in feature_cols:
        if col not in important_features:
            optional_inputs[col] = st.number_input(col, value=0.0)

# ---------------- COMBINE INPUT ----------------
input_data = {}

for col in feature_cols:
    if col in important_features:
        input_data[col] = user_inputs[col]
    else:
        input_data[col] = optional_inputs.get(col, 0.0)

# ---------------- RECOMMENDATION ----------------
if st.button("🔍 Recommend Species"):

    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_cols]

        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df

        prediction = model.predict(input_scaled)

        if label_encoder is not None:
            health_status = label_encoder.inverse_transform(prediction)[0]
        else:
            health_status = str(prediction[0])

        # ---------------- SMART SPECIES LOGIC ----------------
        possible_species = location_species_data[location]
        stress = user_inputs["Env_Stress_Composite"]

        if stress < 0.3:
            recommended_species = possible_species[0]
        elif stress < 0.7:
            recommended_species = possible_species[1]
        else:
            recommended_species = possible_species[-1]

        # ---------------- SURVIVAL RATE ----------------
        try:
            probs = model.predict_proba(input_scaled)
            survival_rate = round(max(probs[0]) * 100, 2)
        except:
            survival_rate = 70

        # ---------------- DONOR SITE ----------------
        donor_site = donor_sites.get(recommended_species, "General Reef")

        # ---------------- OUTPUT ----------------
        st.success("✅ Recommendation Ready!")

        st.subheader("🌿 Recommended Species")
        st.write(f"**{recommended_species}**")

        st.subheader("📊 Estimated Survival Rate")
        st.write(f"**{survival_rate}%**")

        st.subheader("🧠 Predicted Reef Condition")
        st.write(f"**{health_status}**")

        st.subheader("🌍 Suggested Donor Site")
        st.write(f"**{donor_site}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")