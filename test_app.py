import joblib
import pandas as pd
import sys

def test_prediction():
    try:
        print("Loading models and assets...")
        model = joblib.load("XGBoost-tuned-model.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        print("Models loaded successfully.\n")
        
        print("Creating multiple test cases...\n")
        
        test_cases = [
            {
                "name": "Healthy Patient",
                "Age": 40, "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0, "MaxHR": 140, "Oldpeak": 1.0,
                "Sex": "M", "ChestPainType": "ATA", "RestingECG": "Normal", "ExerciseAngina": "N", "ST_Slope": "Flat"
            },
            {
                "name": "High Risk Patient",
                "Age": 65, "RestingBP": 160, "Cholesterol": 300, "FastingBS": 1, "MaxHR": 110, "Oldpeak": 3.0,
                "Sex": "M", "ChestPainType": "ASY", "RestingECG": "LVH", "ExerciseAngina": "Y", "ST_Slope": "Down"
            },
            {
                "name": "Borderline Female Patient",
                "Age": 55, "RestingBP": 140, "Cholesterol": 250, "FastingBS": 0, "MaxHR": 130, "Oldpeak": 1.5,
                "Sex": "F", "ChestPainType": "NAP", "RestingECG": "ST", "ExerciseAngina": "N", "ST_Slope": "Up"
            }
        ]

        for idx, case in enumerate(test_cases, 1):
            print(f"--- Test Case {idx}: {case['name']} ---")
            
            raw_input = {
                "Age": case["Age"],
                "RestingBP": case["RestingBP"],
                "Cholesterol": case["Cholesterol"],
                "FastingBS": case["FastingBS"],
                "MaxHR": case["MaxHR"],
                "Oldpeak": case["Oldpeak"],
                f"Sex_{case['Sex']}": 1,
                f"ChestPainType_{case['ChestPainType']}": 1,
                f"RestingECG_{case['RestingECG']}": 1,
                f"ExerciseAngina_{case['ExerciseAngina']}": 1,
                f"ST_Slope_{case['ST_Slope']}": 1,
            }
            
            input_df = pd.DataFrame([raw_input])
            
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
                    
            input_df = input_df[expected_columns]
            
            numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            prediction = model.predict(input_df)[0]
            
            if prediction == 1:
                print("Prediction: Risk of Heart Stroke Detected! (1)")
            else:
                print("Prediction: No risk of Heart Stroke detected. (0)")
            print("-" * 40 + "\n")
            
        print("✅ Multiple data tests executed successfully.")
        
    except Exception as e:
        print(f"\n❌ Error encountered during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_prediction()
