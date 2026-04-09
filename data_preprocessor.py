import pandas as pd
import numpy as np
import os

def process_weather(file_path):
    print(f"Reading weather data from {file_path}...")
    # Read weather data, skipping the unit row (index 0 after header)
    weather_df = pd.read_excel(file_path, skiprows=[1])
    
    # Ensure numeric types
    cols_to_numeric = ['日最高气温', '日最低气温', '降雨量']
    for c in cols_to_numeric:
        weather_df[c] = pd.to_numeric(weather_df[c], errors='coerce')
        
    weather_df['日平均气温'] = (weather_df['日最高气温'] + weather_df['日最低气温']) / 2.0
    
    # Filter to growing season (April to August)
    growing_df = weather_df[(weather_df['月'] >= 4) & (weather_df['月'] <= 8)]
    
    # Calculate cumulative features (sum over the growing season)
    annual_weather = growing_df.groupby('年份').agg(
        Acc_TMAX=('日最高气温', 'sum'),
        Acc_TMIN=('日最低气温', 'sum'),
        Acc_TAVG=('日平均气温', 'sum'),
        Acc_RAIN=('降雨量', 'sum')
    ).reset_index()
    
    # Convert '年份' to integer
    annual_weather['年份'] = annual_weather['年份'].astype(int)
    
    return annual_weather

def process_sim_data(file_path):
    print(f"Reading simulation data from {file_path}...")
    sim_xl = pd.ExcelFile(file_path)
    merged_data = []
    
    for sheet in sim_xl.sheet_names:
        df = sim_xl.parse(sheet)
        # Remove the 'Mean' row if it exists
        df = df[df['YEAR'] != 'Mean'].copy()
        
        # Extract fertilizer (FERCUM), irrigation (IRCUM), and yield (WRR14) data
        req_cols = ['YEAR', 'FERCUM', 'IRCUM', 'WRR14']
        existing_cols = [c for c in req_cols if c in df.columns]
        df_subset = df[existing_cols].copy()
        
        df_subset['YEAR'] = df_subset['YEAR'].astype(int)
        merged_data.append(df_subset)
        
    final_sim_df = pd.concat(merged_data, ignore_index=True)
    return final_sim_df

def main():
    weather_path = r'D:\learning\modle_Tan\代理模型\梁平站天气数据.xlsx'
    sim_path = r'D:\learning\modle_Tan\代理模型\IR×FER66.xlsx'
    output_path = r'D:\learning\modle_Tan\代理模型\surrogate_dataset.csv'
    
    weather_features = process_weather(weather_path)
    print("Extracted weather features:", weather_features.columns.tolist())
    
    sim_data = process_sim_data(sim_path)
    print("Extracted simulation features:", sim_data.columns.tolist())
    
    print("Merging data...")
    # Merge on YEAR
    merged_df = pd.merge(sim_data, weather_features, left_on='YEAR', right_on='年份', how='inner')
    
    # Drop the duplicate '年份' column
    if '年份' in merged_df.columns:
        merged_df = merged_df.drop(columns=['年份'])
        
    print(f"Saving to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print("Done! Dataset shape:", merged_df.shape)
    print("Final columns:", merged_df.columns.tolist())
    print(merged_df.head())

if __name__ == '__main__':
    main()
