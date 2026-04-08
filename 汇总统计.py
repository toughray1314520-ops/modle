# -*- coding: utf-8 -*-
"""
把所有 summary_*.csv 合并到一个 Excel 文件，方便 Origin 画图
运行前请确保已经执行过主分析代码，生成了 summary_*.csv 文件
"""

import pandas as pd
import os
import glob

# ──────────────── 配置部分（通常不用改） ────────────────
base_dir          = r'D:\learning\modle_Tan\数据分析'                  # 主代码运行目录
output_folder     = r'D:\learning\modle_Tan\数据分析'         # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

output_excel_path = os.path.join(output_folder, '水稻水肥显著性分析汇总_Origin用.xlsx')

indicators = ['WAGT', 'WRR14', 'WUE', 'IWUE', 'ET_WUE', 'NUE', 'SPFERT']

# ──────────────── 主逻辑 ────────────────
writer = pd.ExcelWriter(output_excel_path, engine='openpyxl')

sheet_written = 0

for ind in indicators:
    pattern = os.path.join(base_dir, f'summary_{ind}.csv')
    files = glob.glob(pattern)
    
    if files:
        df = pd.read_csv(files[0], encoding='utf-8-sig')
        # 确保有排序列（如果没有可加）
        if 'Treatment_Combo' in df.columns:
            def sort_key(x):
                if '_' not in str(x): return (99, 0)
                irr, fert = str(x).split('_', 1)
                pct = int(fert.replace('%N','')) if fert.endswith('%N') else 0
                irr_order = {'W1':0, 'W2':1, 'W3':2}
                return (irr_order.get(irr, 99), -pct)
            
            df = df.sort_values(by='Treatment_Combo', key=lambda col: col.map(sort_key))
        
        sheet_name = ind[:31]  # Excel sheet 名长度限制
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"已写入 sheet：{sheet_name}   ({len(df)} 行)")
        sheet_written += 1
    else:
        print(f"未找到：{pattern}")

# ── 额外写入原始数据（可选，但推荐） ──
raw_csv = os.path.join(base_dir, 'raw_data_all.csv')
if os.path.exists(raw_csv):
    try:
        raw = pd.read_csv(raw_csv, encoding='utf-8-sig', low_memory=False)
        keep_cols = ['YEAR', 'Treatment_Combo', 'Irrigation_Level', 'Fertilizer_Level',
                     'Fertilizer_Pct', 'WAGT', 'WRR14', 'WUE', 'IWUE', 'ET_WUE', 'NUE', 'SPFERT']
        keep_cols = [c for c in keep_cols if c in raw.columns]
        raw[keep_cols].to_excel(writer, sheet_name='原始数据关键列', index=False)
        print("已写入 sheet：原始数据关键列")
    except Exception as e:
        print("写入原始数据失败：", e)

# ── 写入使用说明 sheet ──
info = pd.DataFrame({
    '使用说明': [
        '每个 sheet (WAGT / WRR14 / WUE 等) 包含：',
        '  - Treatment_Combo：处理组合（如 W3_100%N）',
        '  - Mean / SD / SEM / n：均值、标准差、标准误、样本数',
        '  - Letter：显著性分组字母（相同字母 = 无显著差异，p≥0.05）',
        '',
        'Origin 推荐作图方式：',
        '1. 柱状图（最常见 SCI 格式）：',
        '   - X = Treatment_Combo',
        '   - Y = Mean',
        '   - Error Bar = SEM',
        '   - 加 Text 层 → 把 Letter 列拖入作为标注',
        '2. 箱线图：直接用 "原始数据关键列" sheet',
        '3. 排序已按 W1→W2→W3 + 氮肥从高到低处理',
        '',
        '如需调整字母分组逻辑或增加其他指标，请告诉我'
    ]
})
info.to_excel(writer, sheet_name='使用说明', index=False)

writer.close()

print(f"\n汇总完成！共写入 {sheet_written} 个指标 sheet")
print("文件位置：")
print(output_excel_path)
print("\n现在可以用 Origin 直接打开这个 Excel 文件开始作图了～")