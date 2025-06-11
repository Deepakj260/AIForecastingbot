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
st.title("AI Forecast Bot: Lifting Quantity Prediction (Part-wise)")

st.markdown("Upload your enriched Excel file (with features like lags, rolling averages, seasonality, etc.)")
uploaded_file = st.file_uploader("Choose a file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Month'] = pd.to_datetime(df['Month'])
    df.sort_values(['Part No', 'Month'], inplace=True)

    search_input = st.text_input("🔍 Search for Part No")
    filtered_parts = df['Part No'].astype(str).unique()
    if search_input:
        filtered_parts = [p for p in filtered_parts if search_input.lower() in str(p).lower()]

    part_selected = st.selectbox("Select a part to forecast", filtered_parts)
    df_part = df[df['Part No'].astype(str) == str(part_selected)].copy()

    # Feature list based on enriched data
    features = [
        'Firm Schedule Qty', 'Firm_Lag1', 'Actual_Lag1', 'Month_Num', 'Year', 'Quarter',
        'Avg_Gap_Percent', 'Seasonal_Firm_Avg', 'Holiday_Impact',
        'Rolling_6mo_Firm', 'Rolling_6mo_Actual'
    ]
    target = 'Actual Lifting Qty'

    df_model = df_part.dropna(subset=features + [target])
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=4)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_results = X_test.copy()
    test_results['Month'] = df_model['Month'].iloc[-4:].values
    test_results['Actual'] = y_test.values
    test_results['Predicted'] = y_pred
    test_results['Error'] = test_results['Actual'] - test_results['Predicted']

    st.subheader("📊 Forecast Results (Last 4 Months)")
    st.dataframe(test_results[['Month', 'Actual', 'Predicted', 'Error']].round(2))
    st.line_chart(test_results.set_index('Month')[['Actual', 'Predicted']])

    # Forecast Firm Schedule Qty for next 24 months
    st.subheader("📈 Forecast Firm Schedule and Lifting for Next 24 Months")
    df_firm = df_part[['Month', 'Firm Schedule Qty']].copy()
    df_firm['Month_Num'] = df_firm['Month'].dt.month
    df_firm['Year'] = df_firm['Month'].dt.year
    firm_features = ['Month_Num', 'Year']
    firm_target = 'Firm Schedule Qty'
    df_firm_model = df_firm.copy()
    X_firm = df_firm_model[firm_features]
    y_firm = df_firm_model[firm_target]
    firm_model = RandomForestRegressor(n_estimators=100, random_state=42)
    firm_model.fit(X_firm, y_firm)

    last_month = df_part['Month'].max()
    future_months = [last_month + relativedelta(months=i+1) for i in range(24)]
    firm_forecast_df = pd.DataFrame({
        'Month': future_months,
        'Month_Num': [m.month for m in future_months],
        'Year': [m.year for m in future_months]
    })
    firm_forecast_df['Firm Schedule Qty'] = firm_model.predict(firm_forecast_df[['Month_Num', 'Year']])

    # Prepare features for lifting prediction
    last_known = df_part.iloc[-1]
    firm_forecast_df['Firm_Lag1'] = last_known['Firm Schedule Qty']
    firm_forecast_df['Actual_Lag1'] = last_known['Actual Lifting Qty']
    firm_forecast_df['Quarter'] = firm_forecast_df['Month'].dt.quarter
    firm_forecast_df['Avg_Gap_Percent'] = last_known['Avg_Gap_Percent']
    firm_forecast_df['Seasonal_Firm_Avg'] = df_part.groupby('Month_Num')['Firm Schedule Qty'].transform('mean').mean()
    firm_forecast_df['Holiday_Impact'] = firm_forecast_df['Month_Num'].isin([8, 9, 10]).astype(int)
    firm_forecast_df['Rolling_6mo_Firm'] = df_part['Firm Schedule Qty'].tail(6).mean()
    firm_forecast_df['Rolling_6mo_Actual'] = df_part['Actual Lifting Qty'].tail(6).mean()

    full_features = firm_forecast_df[features]
    firm_forecast_df['Predicted Lifting Qty'] = model.predict(full_features)

    st.dataframe(firm_forecast_df[['Month', 'Firm Schedule Qty', 'Predicted Lifting Qty']].round(2))

    fig, ax = plt.subplots()
    ax.plot(firm_forecast_df['Month'], firm_forecast_df['Firm Schedule Qty'], label='Forecasted Firm Schedule')
    ax.plot(firm_forecast_df['Month'], firm_forecast_df['Predicted Lifting Qty'], label='Predicted Lifting')
    ax.set_title("24-Month Forecasted Firm Schedule and Lifting")
    ax.set_xlabel("Month")
    ax.set_ylabel("Quantity")
    ax.legend()
    st.pyplot(fig)

    output_combined = BytesIO()
    firm_forecast_df.to_excel(output_combined, index=False)
    output_combined.seek(0)
    st.download_button(
        label="📥 Download 24-Month Forecast (Firm + Lifting)",
        data=output_combined,
        file_name=f"forecasted_firm_lifting_{part_selected}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
