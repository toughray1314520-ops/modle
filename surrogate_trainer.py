import pandas as pd
import numpy as np
import time
import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline

def main():
    # 1. 读取数据
    data_path = r'D:\learning\modle_Tan\代理模型\surrogate_dataset.csv'
    df = pd.read_csv(data_path)
    
    print(f"已加载数据: {data_path}, 形状: {df.shape}")

    # 定义特征和目标变量
    features = ['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']
    target = 'WRR14'

    X = df[features]
    y = df[target]

    # 2. 初始化5折交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 定义评估指标
    scoring = {
        'r2': make_scorer(r2_score),
        'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
    }

    # 3. 定义模型
    # 树模型
    tree_models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    }

    # 高斯过程模型
    # 为了防止数据泄露，在交叉验证中使用 Pipeline 包含 StandardScaler 和 GPR
    # 使用 RBF 叠加 WhiteKernel 来处理数据中的潜在噪声，同时放宽边界避免优化器不收敛
    from sklearn.gaussian_process.kernels import WhiteKernel
    kernel = C(1.0, (1e-4, 1e4)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0, random_state=42, normalize_y=True)
    gpr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', gpr)
    ])

    print("\n--- 开始5折交叉验证 ---")
    
    best_tree_name = None
    best_tree_score = -np.inf

    # 4. 评估树模型
    for name, model in tree_models.items():
        print(f"\n正在评估 {name}...")
        start_time = time.time()
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        avg_r2 = np.mean(cv_results['test_r2'])
        avg_rmse = np.mean(cv_results['test_rmse'])
        print(f"[{name}] 5折平均 R2: {avg_r2:.4f}, 平均 RMSE: {avg_rmse:.4f}, 耗时: {time.time()-start_time:.2f}s")
        
        # 记录最佳树模型 (根据 R2)
        if avg_r2 > best_tree_score:
            best_tree_score = avg_r2
            best_tree_name = name

    # 5. 评估 GaussianProcessRegressor
    print(f"\n正在评估 GaussianProcessRegressor...")
    start_time = time.time()
    cv_results_gpr = cross_validate(gpr_pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    avg_r2_gpr = np.mean(cv_results_gpr['test_r2'])
    avg_rmse_gpr = np.mean(cv_results_gpr['test_rmse'])
    print(f"[GaussianProcessRegressor] 5折平均 R2: {avg_r2_gpr:.4f}, 平均 RMSE: {avg_rmse_gpr:.4f}, 耗时: {time.time()-start_time:.2f}s")

    # 6. 在全量数据上训练最终模型并保存
    print(f"\n--- 最佳树模型为 {best_tree_name} (CV 平均 R2: {best_tree_score:.4f}) ---")
    
    # 训练并保存最佳树模型
    print(f"正在全量数据上训练 {best_tree_name}...")
    best_tree_model = tree_models[best_tree_name]
    best_tree_model.fit(X, y)
    
    best_tree_path = r'D:\learning\modle_Tan\代理模型\surrogate_model_best_tree.pkl'
    joblib.dump(best_tree_model, best_tree_path)
    print(f"最佳树模型已保存至: {best_tree_path}")

    # 训练并保存 GPR 及其 StandardScaler
    print(f"\n正在全量数据上训练 GaussianProcessRegressor 及其 StandardScaler...")
    
    # 单独实例化 StandardScaler 并进行拟合，以便单独保存
    scaler_gpr = StandardScaler()
    X_scaled = scaler_gpr.fit_transform(X)
    
    # 在全量数据上重新训练 GPR
    gpr_final = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0, random_state=42, normalize_y=True)
    gpr_final.fit(X_scaled, y)
    
    gpr_model_path = r'D:\learning\modle_Tan\代理模型\surrogate_model_gpr.pkl'
    scaler_path = r'D:\learning\modle_Tan\代理模型\scaler_gpr.pkl'
    
    joblib.dump(gpr_final, gpr_model_path)
    joblib.dump(scaler_gpr, scaler_path)
    print(f"GPR 模型已保存至: {gpr_model_path}")
    print(f"GPR StandardScaler 已保存至: {scaler_path}")

if __name__ == "__main__":
    main()
