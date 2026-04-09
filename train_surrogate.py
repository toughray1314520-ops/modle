import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("1. 读取气象数据并提取生育期(4-8月)特征...")
weather_df = pd.read_excel('D:\learning\modle_Tan\代理模型\梁平站天气数据.xlsx', skiprows=[1])
cols_to_numeric = ['日最高气温', '日最低气温', '日照时数', '平均风速', '降雨量', '实际水汽压']
for c in cols_to_numeric:
    weather_df[c] = pd.to_numeric(weather_df[c], errors='coerce')

weather_df['日平均气温'] = (weather_df['日最高气温'] + weather_df['日最低气温']) / 2.0

# Filter to growing season (April to August)
growing_df = weather_df[(weather_df['月'] >= 4) & (weather_df['月'] <= 8)]

# Aggregate to annual level
annual_weather = growing_df.groupby('年份').agg(
    GS_TMAX_mean=('日最高气温', 'mean'),
    GS_TMIN_mean=('日最低气温', 'mean'),
    GS_TAVG_mean=('日平均气温', 'mean'),
    GS_RAIN_sum=('降雨量', 'sum'),
    GS_SUN_sum=('日照时数', 'sum'),
    GS_WIND_mean=('平均风速', 'mean'),
    GS_VAPOR_mean=('实际水汽压', 'mean')
).reset_index()

print("2. 读取模拟数据并融合(66个水肥组合)...")
sim_xl = pd.ExcelFile('D:\learning\modle_Tan\代理模型\IR×FER66.xlsx')
merged_data = []

for sheet in sim_xl.sheet_names:
    df = sim_xl.parse(sheet)
    # Remove the 'Mean' row
    df = df[df['YEAR'] != 'Mean'].copy()
    df['YEAR'] = df['YEAR'].astype(int)
    # Merge with weather features
    merged = pd.merge(df, annual_weather, left_on='YEAR', right_on='年份', how='inner')
    merged_data.append(merged)

final_df = pd.concat(merged_data, ignore_index=True)
final_df.to_csv('D:\learning\modle_Tan\代理模型\surrogate_dataset.csv', index=False)
print(f"融合数据集已保存至: D:\learning\modle_Tan\代理模型\surrogate_dataset.csv (形状: {final_df.shape})")

print("3. 训练代理模型基线(Random Forest预测WRR14产量)...")
features = ['GS_TMAX_mean', 'GS_TMIN_mean', 'GS_TAVG_mean', 'GS_RAIN_sum', 
            'GS_SUN_sum', 'GS_WIND_mean', 'GS_VAPOR_mean', 'IRCUM', 'FERCUM']
target = 'WRR14'

X = final_df[features]
y = final_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"测试集 R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"测试集 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} kg/ha")

print("\n模型特征重要性 (Feature Importance):")
importances = rf.feature_importances_
for f, imp in zip(features, importances):
    print(f"- {f}: {imp:.4f}")
