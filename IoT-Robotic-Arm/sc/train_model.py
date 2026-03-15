import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

def main():
    # 1. Load the data
    data_path = r"d:\RoboticArm\IOT-Robotic-Arm\IoT-Robotic-Arm\scripts\saved_positions.json"
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return
        
    df = pd.DataFrame(data)
    
    # 2. Feature Engineering based on the "position difference" concept
    # Extract positional difference (pixels)
    df['pos_diff'] = df['clicked_x'] - df['center_x']
    
    # Extract angle difference (degrees)
    df['angle_diff'] = df['saved_s1_angle'] - df['initial_s1_angle']
    
    X = df[['pos_diff']]
    y = df['angle_diff']
    
    # 3. Split the data (95% training, 5% testing)
    # Using test_size=0.05 to leave 95% of the data for training, as requested.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)} (95%)")
    print(f"Testing samples: {len(X_test)} (5%)")
    
    # 4. Model Definition & Training
    print("\nTraining Polynomial Regression Model (Degree=2)...")
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print("Training Data Metrics:")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"  MAE (Mean Absolute Error): {mean_absolute_error(y_train, y_pred_train):.4f} degrees")
    print(f"  R2 Score: {r2_score(y_train, y_pred_train):.4f}")
    
    print("\nTesting Data Metrics (Unseen Data):")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print(f"  MAE (Mean Absolute Error): {mean_absolute_error(y_test, y_pred_test):.4f} degrees")
    print(f"  R2 Score: {r2_score(y_test, y_pred_test):.4f}")
    
    # 6. Plotting model predictions vs actual data
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))
    
    # Scatter the actual data points
    plt.scatter(X_train['pos_diff'], y_train, color='blue', label='Actual Data (Train)', alpha=0.7)
    plt.scatter(X_test['pos_diff'], y_test, color='red', label='Actual Data (Test)', marker='x', s=100, alpha=0.9)
    
    # Generate smooth curve for model predictions
    x_range = np.linspace(df['pos_diff'].min() - 20, df['pos_diff'].max() + 20, 200).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=['pos_diff'])
    y_range_pred = model.predict(x_range_df)
    
    plt.plot(x_range, y_range_pred, color='green', linewidth=2, label='Model Prediction (Degree=2)')
    
    plt.title('Position Difference vs. Angle Difference')
    plt.xlabel('Position Difference (pixels)')
    plt.ylabel('Angle Difference (degrees)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save plot to scripts directory
    plot_save_path = os.path.join(os.path.dirname(data_path), "model_predictions_plot.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully at: {plot_save_path}")
    
    # 7. Save the Model
    model_dir = r"d:\RoboticArm\IOT-Robotic-Arm\IoT-Robotic-Arm\models"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "position_to_angle_model.pkl")
    
    joblib.dump(model, model_save_path)
    print(f"\nModel saved successfully at: {model_save_path}")

if __name__ == "__main__":
    main()
