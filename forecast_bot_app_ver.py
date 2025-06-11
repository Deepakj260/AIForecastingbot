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

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    test_results = X_test.copy()
    test_results['Month'] = df_model['Month'].iloc[-4:].values
    test_results['Actual'] = y_test.values
    test_results['Predicted'] = y_pred
    test_results['Error'] = test_results['Actual'] - test_results['Predicted']

    st.subheader("📊 Forecast Results (Last 4 Months)")
    st.dataframe(test_results[['Month', 'Actual', 'Predicted', 'Error']].round(2))

    # Line Chart
    st.line_chart(test_results.set_index('Month')[['Actual', 'Predicted']])

    # Forecast for next 24 months using CAGR
    st.subheader("📈 24-Month Forecast with CAGR Growth")
    last_month = df_part['Month'].max()
    base_row = df_part.iloc[-1].copy()
    firm_series = df_part['Firm Schedule Qty']
    months_elapsed = (df_part['Month'].max().year - df_part['Month'].min().year) * 12 + (df_part['Month'].max().month - df_part['Month'].min().month)
    cagr = ((firm_series.iloc[-1] / firm_series.iloc[0]) ** (1 / (months_elapsed / 12))) - 1 if months_elapsed > 0 else 0

    future_months = [last_month + relativedelta(months=i+1) for i in range(24)]
    future_rows = []
    for i, month in enumerate(future_months):
        row = base_row.copy()
        row['Month'] = month
        row['Month_Num'] = month.month
        row['Year'] = month.year
        row['Quarter'] = (month.month - 1) // 3 + 1
        row['Firm_Lag1'] = row['Firm Schedule Qty'] * ((1 + cagr) ** 1)
        row['Actual_Lag1'] = row['Actual Lifting Qty']
        row['Avg_Gap_Percent'] = row['Avg_Gap_Percent']
        row['Seasonal_Firm_Avg'] = df_part[df_part['Month_Num'] == month.month]['Firm Schedule Qty'].mean()
        row['Holiday_Impact'] = 1 if month.month in [8, 9, 10] else 0
        row['Rolling_6mo_Firm'] = df_part['Firm Schedule Qty'].tail(6).mean()
        row['Rolling_6mo_Actual'] = df_part['Actual Lifting Qty'].tail(6).mean()
        row['Firm Schedule Qty'] = row['Firm_Lag1']
        future_rows.append(row[features + ['Month', 'Part No']])

    future_df = pd.DataFrame(future_rows)
    future_df['Part No'] = part_selected
    future_df['Predicted Lifting'] = model.predict(future_df[features])

    st.dataframe(future_df[['Month', 'Part No', 'Firm Schedule Qty', 'Predicted Lifting']].round(2))

    # Line chart of 24-month forecast
    fig, ax = plt.subplots()
    ax.plot(future_df['Month'], future_df['Firm Schedule Qty'], label='Firm Schedule (CAGR-based)')
    ax.plot(future_df['Month'], future_df['Predicted Lifting'], label='Predicted Lifting')
    ax.set_title("24-Month Forecast with CAGR Growth")
    ax.set_xlabel("Month")
    ax.set_ylabel("Quantity")
    ax.legend()
    st.pyplot(fig)

    # Downloadable output
    output = BytesIO()
    future_df.to_excel(output, index=False)
    output.seek(0)
    st.download_button(
        label="📥 Download 24-Month Forecast",
        data=output,
        file_name=f"forecast_24months_CAGR_{part_selected}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
