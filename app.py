"""
BigMart Sales Prediction — Streamlit App
Run: streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/sales_model.pkl", "rb") as f:
        return pickle.load(f)

artifacts      = load_artifacts()
model          = artifacts["model"]
label_encoders = artifacts["label_encoders"]
feature_cols   = artifacts["feature_cols"]
cat_cols       = artifacts["cat_cols"]

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_input(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    fat_map = {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(fat_map)

    df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]
    df["Item_Category"] = df["Item_Identifier"].str[:2].map(
        {"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}
    ).fillna("Food")
    df.loc[df["Item_Category"] == "Non-Consumable", "Item_Fat_Content"] = "Non-Edible"
    df["Price_per_Weight"]     = df["Item_MRP"] / df["Item_Weight"]
    df["Visibility_MeanRatio"] = 1.0

    df = df.drop(columns=["Item_Identifier", "Outlet_Identifier",
                            "Outlet_Establishment_Year"], errors="ignore")

    for col in cat_cols:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛒 BigMart Sales Predictor")
st.markdown("Enter product and outlet details below to predict expected sales.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📦 Item Details")
        item_id      = st.text_input("Item Identifier", value="FD_X01")
        item_weight  = st.number_input("Item Weight (kg)", 0.0, 50.0, 9.3, 0.1)
        item_fat     = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        item_vis     = st.slider("Item Visibility", 0.0, 0.35, 0.016, 0.001)
        item_type    = st.selectbox("Item Type", [
            "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy",
            "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
            "Health and Hygiene", "Household", "Meat", "Others",
            "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
        ])
        item_mrp     = st.number_input("Item MRP (₹)", 10.0, 500.0, 107.86, 0.5)

    with col2:
        st.subheader("🏪 Outlet Details")
        outlet_id    = st.text_input("Outlet Identifier", value="OUT049")
        outlet_year  = st.number_input("Outlet Establishment Year", 1980, 2015, 1999, 1)
        outlet_size  = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_loc   = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_type  = st.selectbox("Outlet Type", [
            "Grocery Store", "Supermarket Type1",
            "Supermarket Type2", "Supermarket Type3",
        ])

    with col3:
        st.subheader("📊 Prediction")
        submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)

        if submitted:
            input_dict = {
                "Item_Identifier":           item_id,
                "Item_Weight":               item_weight,
                "Item_Fat_Content":          item_fat,
                "Item_Visibility":           item_vis,
                "Item_Type":                 item_type,
                "Item_MRP":                  item_mrp,
                "Outlet_Identifier":         outlet_id,
                "Outlet_Establishment_Year": outlet_year,
                "Outlet_Size":               outlet_size,
                "Outlet_Location_Type":      outlet_loc,
                "Outlet_Type":               outlet_type,
            }
            try:
                X    = preprocess_input(input_dict)
                pred = float(model.predict(X)[0])
                st.metric("Predicted Sales", f"₹ {pred:,.2f}")
                st.success(f"Estimated annual sales: **₹ {pred:,.0f}**")

                # Rough category
                if pred < 1500:
                    st.info("📉 Below-average performer")
                elif pred < 3500:
                    st.warning("📊 Average performer")
                else:
                    st.success("📈 High performer")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("BigMart Sales Prediction | Model: XGBoost / GradientBoosting")
