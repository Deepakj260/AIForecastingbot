
import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from generate_forecast import generate_stable_forecast

st.set_page_config(page_title="AI Forecast Bot", layout="wide")
st.title("📈 AI Forecast Bot - 24 Month Stable Forecast")

# Upload historical Excel file
uploaded_file = st.file_uploader("Upload historical firm & actual lifting Excel file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    forecast_df = generate_stable_forecast(df)

    st.success("✅ Forecast generated successfully!")
    st.dataframe(forecast_df)

    # Graph per part
    selected_part = st.selectbox("📊 Select Part No to View Graph", forecast_df['Part No'].unique())
    df_part = forecast_df[forecast_df['Part No'] == selected_part]

    fig, ax = plt.subplots()
    ax.plot(df_part['Month'], df_part['Stable Forecast Qty (Seasonal Overwrite)'], marker='o', linestyle='-', label='Forecast Qty')
    ax.set_title(f"24-Month Forecast - Part No: {selected_part}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Forecasted Qty")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Download forecast
    buffer = BytesIO()
    forecast_df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="📥 Download 24-Month Forecast Excel",
        data=buffer,
        file_name="Stable_24_Month_Forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("📂 Please upload a valid Excel file with 'Part No', 'Month', and 'Actual Lifting Qty'")
