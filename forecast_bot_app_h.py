import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="AI Forecast Bot", layout="wide")
st.title("AI Forecast Bot: Actual Lifting Prediction Based on Uploaded Firm Schedule")

# Step 1: Upload historical data once and store in session
st.markdown("Upload your Excel file with at least 2 years of firm + lifting history (only once)")
if 'history_df' not in st.session_state:
    hist_file = st.file_uploader("Upload history file (only first time)", type="xlsx", key='hist_upload')
    if hist_file:
        st.session_state['history_df'] = pd.read_excel(hist_file)

if 'history_df' in st.session_state:
    df = st.session_state['history_df']
    df['Month'] = pd.to_datetime(df['Month'])
    df.sort_values(['Part No', 'Month'], inplace=True)

    search_input = st.text_input("🔍 Search for Part No")
    filtered_parts = df['Part No'].astype(str).unique()
    if search_input:
        filtered_parts = [p for p in filtered_parts if search_input.lower() in str(p).lower()]

    part_selected = st.selectbox("Select a part to forecast", filtered_parts)
    df_part = df[df['Part No'].astype(str) == str(part_selected)].copy()

    # Feature engineering
    df_part['Month_Num'] = df_part['Month'].dt.month
    df_part['Year'] = df_part['Month'].dt.year
    df_part['Quarter'] = df_part['Month'].dt.quarter
    df_part['Firm_Lag1'] = df_part['Firm Schedule Qty'].shift(1)
    df_part['Actual_Lag1'] = df_part['Actual Lifting Qty'].shift(1)
    df_part['Avg_Gap_Percent'] = (df_part['Firm Schedule Qty'] - df_part['Actual Lifting Qty']) / df_part['Firm Schedule Qty']
    df_part['Rolling_6mo_Firm'] = df_part['Firm Schedule Qty'].rolling(6).mean()
    df_part['Rolling_6mo_Actual'] = df_part['Actual Lifting Qty'].rolling(6).mean()
    df_part['Holiday_Impact'] = df_part['Month_Num'].isin([8, 9, 10]).astype(int)
    df_part['Seasonal_Firm_Avg'] = df_part.groupby('Month_Num')['Firm Schedule Qty'].transform('mean')

    features = [
        'Firm Schedule Qty', 'Firm_Lag1', 'Actual_Lag1', 'Month_Num', 'Year', 'Quarter',
        'Avg_Gap_Percent', 'Seasonal_Firm_Avg', 'Holiday_Impact',
        'Rolling_6mo_Firm', 'Rolling_6mo_Actual'
    ]
    target = 'Actual Lifting Qty'

    df_model = df_part.dropna(subset=features + [target])
    X = df_model[features]
    y = df_model[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Step 2: Upload new firm schedule for prediction only
    st.markdown("---")
    st.markdown("📥 Upload future firm schedule to predict actual lifting")
    future_file = st.file_uploader("Upload future firm schedule", type="xlsx", key='future_upload')

    if future_file:
        df_future = pd.read_excel(future_file)
        df_future['Month'] = pd.to_datetime(df_future['Month'])

        if 'Part No' not in df_future.columns:
            st.error("❌ 'Part No' column is missing in uploaded file.")
            st.stop()

        if str(part_selected) not in df_future['Part No'].astype(str).unique():
            st.warning(f"⚠️ Part No '{part_selected}' not found in uploaded future schedule. Showing data for other available parts.")
            available_parts = df_future['Part No'].astype(str).unique()
            df_future = df_future[df_future['Part No'].astype(str).isin(available_parts)].copy()
        else:
            df_future = df_future[df_future['Part No'].astype(str) == str(part_selected)].copy()

        df_future['Month_Num'] = df_future['Month'].dt.month
        df_future['Year'] = df_future['Month'].dt.year
        df_future['Quarter'] = df_future['Month'].dt.quarter
        last_known = df_part.iloc[-1]

        df_future['Firm_Lag1'] = last_known['Firm Schedule Qty']
        df_future['Actual_Lag1'] = last_known['Actual Lifting Qty']
        df_future['Avg_Gap_Percent'] = last_known['Avg_Gap_Percent']
        df_future['Seasonal_Firm_Avg'] = df_part.groupby('Month_Num')['Firm Schedule Qty'].transform('mean').mean()
        df_future['Holiday_Impact'] = df_future['Month_Num'].isin([8, 9, 10]).astype(int)
        df_future['Rolling_6mo_Firm'] = df_part['Firm Schedule Qty'].tail(6).mean()
        df_future['Rolling_6mo_Actual'] = df_part['Actual Lifting Qty'].tail(6).mean()

        X_future = df_future[features]
        if X_future.isnull().any().any():
            st.error("❌ Missing values detected in prediction input. Please check uploaded file.")
            st.dataframe(X_future)
            st.stop()

        df_future['Predicted Lifting Qty'] = model.predict(X_future)

        st.subheader("📊 Predicted Actual Lifting for Uploaded Firm Schedule")
        st.dataframe(df_future[['Month', 'Firm Schedule Qty', 'Predicted Lifting Qty']].round(2))

        fig, ax = plt.subplots()
        ax.plot(df_future['Month'], df_future['Firm Schedule Qty'], label='Firm Schedule')
        ax.plot(df_future['Month'], df_future['Predicted Lifting Qty'], label='Predicted Lifting')
        ax.set_title("Predicted Actual Lifting Based on Uploaded Schedule")
        ax.set_xlabel("Month")
        ax.set_ylabel("Quantity")
        ax.legend()
        st.pyplot(fig)

        output_combined = BytesIO()
        df_future.to_excel(output_combined, index=False)
        output_combined.seek(0)
        st.download_button(
            label="📥 Download Prediction",
            data=output_combined,
            file_name=f"predicted_lifting_{part_selected}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👆 Please upload your history file to begin. It will be cached for this session.")
