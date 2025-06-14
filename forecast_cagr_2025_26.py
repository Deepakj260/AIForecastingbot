
import pandas as pd
from dateutil.relativedelta import relativedelta

def forecast_2025_26_from_cagr(df, cagr_df):
    df['Month'] = pd.to_datetime(df['Month'])
    df['Fiscal_Year'] = df['Month'].apply(lambda d: f"{d.year}-{d.year+1}" if d.month >= 4 else f"{d.year-1}-{d.year}")
    forecast_months = pd.date_range(start='2025-04-01', end='2026-03-31', freq='MS')

    forecast_data = []
    for _, row in cagr_df.iterrows():
        part_no = row['Part No']
        cagr = row['CAGR_2023_2025 (%)']
        if pd.isna(cagr):
            continue

        # Get the latest monthly average from FY 2024-25
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

    return pd.DataFrame(forecast_data)
