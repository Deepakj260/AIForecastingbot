
import pandas as pd
from dateutil.relativedelta import relativedelta

def generate_stable_forecast(df, months_forward=24, min_forecast=10, growth_cap=0.5):
    df['Month'] = pd.to_datetime(df['Month'])
    df.sort_values(['Part No', 'Month'], inplace=True)
    part_numbers = df['Part No'].unique()
    all_part_forecasts = []

    for part_no in part_numbers:
        df_part = df[df['Part No'] == part_no].copy()
        df_part['Actual Lifting Qty'] = df_part['Actual Lifting Qty'].fillna(0)

        if df_part.empty or df_part['Actual Lifting Qty'].sum() == 0:
            continue

        df_part['MoM Change %'] = df_part['Actual Lifting Qty'].pct_change() * 100
        valid_growth = df_part['MoM Change %'].dropna().tail(6)
        avg_monthly_growth = valid_growth.mean() / 100 if not valid_growth.empty else 0
        avg_monthly_growth = min(avg_monthly_growth, growth_cap)

        apr_jun_2024 = df_part[
            (df_part['Month'].dt.year == 2024) &
            (df_part['Month'].dt.month.isin([4, 5, 6]))
        ][['Month', 'Actual Lifting Qty']].copy()
        seasonal_map = apr_jun_2024.set_index(apr_jun_2024['Month'].dt.month)['Actual Lifting Qty'].to_dict()

        latest_value = df_part['Actual Lifting Qty'].iloc[-1]
        current_month = df_part['Month'].max()

        for i in range(months_forward):
            next_month = current_month + relativedelta(months=1)
            month_num = next_month.month

            if month_num in seasonal_map:
                forecast_qty = seasonal_map[month_num]
            else:
                forecast_qty = latest_value * (1 + avg_monthly_growth)

            forecast_qty = min(forecast_qty, latest_value * 10)
            forecast_qty = max(forecast_qty, min_forecast)

            all_part_forecasts.append({
                'Part No': part_no,
                'Month': next_month,
                'Stable Forecast Qty (Seasonal Overwrite)': round(forecast_qty)
            })

            latest_value = forecast_qty
            current_month = next_month

    return pd.DataFrame(all_part_forecasts)
