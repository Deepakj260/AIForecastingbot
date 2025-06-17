
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="AI Forecast Bot", layout="wide")
st.title("📈 AI Forecast Bot with CAGR-based Prediction")

# File upload
hist_file = st.file_uploader("📤 Upload 2-Year History File (Firm + Lifting)", type="xlsx")
if hist_file:
    df = pd.read_excel(hist_file)
    df['Month'] = pd.to_datetime(df['Month'])
    df.sort_values(['Part No', 'Month'], inplace=True)

    # Assign fiscal year (Apr–Mar)
    df['Fiscal_Year'] = df['Month'].apply(lambda d: f"{d.year}-{d.year+1}" if d.month >= 4 else f"{d.year-1}-{d.year}")

    # Compute fiscal totals and CAGR for each part
    fy_totals = df[df['Fiscal_Year'].isin(['2023-2024', '2024-2025'])].groupby(['Part No', 'Fiscal_Year'])['Actual Lifting Qty'].sum().unstack()

    def calculate_two_year_cagr(start, end):
        if pd.notnull(start) and start > 0 and pd.notnull(end):
            return round((((end / start) ** (1 / 2)) - 1) * 100, 2)
        return None

    fy_totals['CAGR_2023_2025 (%)'] = fy_totals.apply(
        lambda row: calculate_two_year_cagr(row.get('2023-2024'), row.get('2024-2025')),
        axis=1
    )

    st.success("✅ History file loaded. CAGR calculated for all parts.")

    # Run forecast based on CAGR
    forecast_months = pd.date_range(start='2025-04-01', end='2026-03-31', freq='MS')
    forecast_data = []

    for _, row in fy_totals.reset_index().iterrows():
        part_no = row['Part No']
        cagr = row['CAGR_2023_2025 (%)']
        if pd.isna(cagr):
            continue

        latest_df = df[(df['Part No'] == part_no) & (df['Fiscal_Year'] == '2024-2025')]
        if latest_df.empty:
            continue

        monthly_avg = latest_df['Actual Lifting Qty'].mean()
        if monthly_avg == 0 or pd.isna(monthly_avg):
            continue

        monthly_growth = (1 + (cagr / 100)) ** (1 / 12)
        forecast_qty = monthly_avg

        for month in forecast_months:
            forecast_data.append({
                'Part No': part_no,
                'Month': month,
                'Forecasted Qty (2025–26)': round(forecast_qty)
            })
            forecast_qty *= monthly_growth

    forecast_df = pd.DataFrame(forecast_data)

    # Part selection
    part_list = sorted(forecast_df['Part No'].unique())
    part_selected = st.selectbox("🔎 Select a Part No to View Forecast", part_list)

    part_hist = df[df['Part No'] == part_selected][['Month', 'Actual Lifting Qty']].copy()
    part_fore = forecast_df[forecast_df['Part No'] == part_selected][['Month', 'Forecasted Qty (2025–26)']].copy()

    # Display table
    st.subheader(f"📄 Forecast for Part No: {part_selected}")
    st.dataframe(part_fore)

    # Display graph
    st.subheader("📊 Forecast Visualization")
    fig, ax = plt.subplots()
    ax.plot(part_hist['Month'], part_hist['Actual Lifting Qty'], label='Historical Actual')
    ax.plot(part_fore['Month'], part_fore['Forecasted Qty (2025–26)'], label='Forecasted Qty (2025–26)', linestyle='--')
    ax.set_xlabel("Month")
    ax.set_ylabel("Quantity")
    ax.legend()
    st.pyplot(fig)

    # Download button
    st.markdown("### 📥 Download Forecast")
    final_export = pd.concat([part_hist.rename(columns={'Actual Lifting Qty': 'Qty'}),
                              part_fore.rename(columns={'Forecasted Qty (2025–26)': 'Qty'})])
    final_export = final_export.sort_values('Month')
    final_export['Part No'] = part_selected
    download = final_export[['Part No', 'Month', 'Qty']]
    st.download_button(
        "Download Excel",
        data=download.to_csv(index=False),
        file_name=f"Forecast_{part_selected}.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Upload your historical firm and lifting data to get started.")
