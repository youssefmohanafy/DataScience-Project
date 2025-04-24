import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the raw model only (no preprocessor)
model = joblib.load("model2_linear_regression.pkl")

# Load data for visuals
df = pd.read_csv("E-commerce_Dataset_cleaned.csv")

# Title
st.title("E-Commerce Profit Predictor")

# EDA plot
st.subheader("Discount vs Sales")
fig = px.scatter(df, x="Discount", y="Sales", color="Product_Category", template="plotly_dark")
st.plotly_chart(fig)

# Sidebar inputs
st.sidebar.header("Order Input")
sales = st.sidebar.number_input("Sales", 0.0)
quantity = st.sidebar.number_input("Quantity", 1, step=1)
discount = st.sidebar.slider("Discount", 0.0, 0.5, 0.1)

# Predict
if st.button("Predict Profit"):
    input_df = pd.DataFrame([[sales, quantity, discount]], columns=['Sales', 'Quantity', 'Discount'])
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Profit: ${prediction:.2f}")
