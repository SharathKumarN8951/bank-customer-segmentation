import streamlit as st
import numpy as np
import joblib

# Load saved scaler, PCA, and model

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("kmeans_model.pkl")

# Page config
st.set_page_config(
    page_title="Bank Customer Segmentation",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Segmentation")
st.subheader("PCA + KMeans Clustering")
st.write("Enter customer details to identify the customer segment")

st.divider()

# -------------------------------------------------
# User Input Fields (ORIGINAL FEATURES – NOT REDUCED)
# -------------------------------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=5)
income = st.number_input("Annual Income ($K)", min_value=0, value=50)
family = st.number_input("Family Size", min_value=1, max_value=6, value=2)
ccavg = st.number_input("Average Credit Card Spend", min_value=0.0, value=1.5)
education = st.selectbox("Education Level", [1, 2, 3])
mortgage = st.number_input("Mortgage Amount", min_value=0, value=0)
securities = st.selectbox("Securities Account", [0, 1])
cd = st.selectbox("CD Account", [0, 1])
online = st.selectbox("Online Banking", [0, 1])

st.divider()

# -------------------------------------------------
# Cluster → Business Meaning Mapping
# (Adjust names if needed after profiling)
# -------------------------------------------------
segment_map = {
    0: "Low-Value Traditional Customers",
    1: "High Net-Worth Customers",
    2: "Young Digital Customers",
    3: "Affluent Online Customers"
}

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if st.button("Predict Customer Segment", use_container_width=True):

    # Combine inputs
    input_data = np.array([[
        age,
        experience,
        income,
        family,
        ccavg,
        education,
        mortgage,
        securities,
        cd,
        online
    ]])

    # Step 1: Scale
    scaled_data = scaler.transform(input_data)

    # Step 2: PCA transform
    pca_data = pca.transform(scaled_data)

    # Step 3: Predict cluster
    cluster = model.predict(pca_data)[0]

    # Display result
    st.success(
        f"Customer Segment: **{segment_map.get(cluster, 'Unknown Segment')}**"
    )

    st.caption(f"(Internal Cluster ID: {cluster})")
