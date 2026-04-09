import joblib
import pandas as pd
import numpy as np

def main():
    # Load Best Tree Model
    tree_model_path = r'D:\learning\modle_Tan\代理模型\surrogate_model_best_tree.pkl'
    print(f"Loading Tree model from {tree_model_path}...")
    tree_model = joblib.load(tree_model_path)
    
    # Load GPR Model and Scaler
    gpr_model_path = r'D:\learning\modle_Tan\代理模型\surrogate_model_gpr.pkl'
    scaler_path = r'D:\learning\modle_Tan\代理模型\scaler_gpr.pkl'
    print(f"Loading GPR model from {gpr_model_path}...")
    gpr_model = joblib.load(gpr_model_path)
    print(f"Loading GPR scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    
    # Mock input array based on the prompt's specified order:
    # [Acc_TMAX, Acc_TMIN, Acc_TAVG, Acc_RAIN, IRCUM, FERCUM]
    mock_input_array = [4319.0, 2988.0, 3650.0, 300.0, 200.0, 190.0]
    Acc_TMAX, Acc_TMIN, Acc_TAVG, Acc_RAIN, IRCUM, FERCUM = mock_input_array
    
    # The model was trained with features: ['FERCUM', 'IRCUM', 'Acc_TMAX', 'Acc_TMIN', 'Acc_TAVG', 'Acc_RAIN']
    # We construct a DataFrame to ensure the feature names and order match what the model expects
    input_df = pd.DataFrame([{
        'FERCUM': FERCUM,
        'IRCUM': IRCUM,
        'Acc_TMAX': Acc_TMAX,
        'Acc_TMIN': Acc_TMIN,
        'Acc_TAVG': Acc_TAVG,
        'Acc_RAIN': Acc_RAIN
    }])
    
    print(f"Input Features:\n{input_df}")
    
    # Perform prediction with Tree Model
    tree_prediction = tree_model.predict(input_df)
    
    # Perform prediction with GPR Model
    # GPR typically requires scaled input
    input_scaled = scaler.transform(input_df)
    gpr_mean, gpr_std = gpr_model.predict(input_scaled, return_std=True)
    
    print(f"\n--- Prediction Results ---")
    print(f"Mock input array [Acc_TMAX, Acc_TMIN, Acc_TAVG, Acc_RAIN, IRCUM, FERCUM]: {mock_input_array}")
    print(f"Tree Predicted Yield Value (WRR14): {tree_prediction[0]}")
    print(f"GPR Predicted Yield Value (WRR14): Mean = {gpr_mean[0]}, Std = {gpr_std[0]}")

if __name__ == "__main__":
    main()
