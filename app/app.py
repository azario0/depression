from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import joblib

# Load the original DataFrame to get unique values
df = pd.read_csv('DepressionStudentDataset.csv')  # Replace with your actual dataset path

def retriev_savings():
    """
    Classifies new data using the saved model, scaler, and label encoders.

    Args:
        new_data (pandas.DataFrame): A DataFrame containing the new data to be classified.

    Returns:
        list: A list of predicted class labels.
    """

    # Load the saved objects

    model = joblib.load('best_classifier_model.joblib')
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    return  model , label_encoders

# Initialize the model predictor
predictor , label_encoders  = retriev_savings()

categorical_cols = df.drop(columns=['Depression']).select_dtypes(include=['object']).columns

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    """
    Render the main prediction page
    """
    unique_values = {col: list(df[col].unique()) for col in categorical_cols}
    
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making depression predictions
    """
    try:
        # Get input data from form
        input_data = {
            'Gender': request.form.get('gender'),
            'Age': int(request.form.get('age')),
            'Academic Pressure': float(request.form.get('academic_pressure')),
            'Study Satisfaction': float(request.form.get('study_satisfaction')),
            'Sleep Duration': request.form.get('sleep_duration'),
            'Dietary Habits': request.form.get('dietary_habits'),
            'Have you ever had suicidal thoughts ?': request.form.get('suicidal_thoughts'),
            'Study Hours': int(request.form.get('study_hours')),
            'Financial Stress': int(request.form.get('financial_stress')),
            'Family History of Mental Illness': request.form.get('family_history')
        }
        input_df = pd.DataFrame(input_data,index=[1])
        
        for col in categorical_cols:
            if col in label_encoders:
                mapping = label_encoders[col]
                
                # Map values using the stored dictionary
                input_df[col] = mapping.transform(input_df[col])
                print(input_df)
                
                # Check for unmapped values and handle them if necessary
                if input_df[col].isnull().any():
                    print(f"Warning: Unmapped values found in column '{col}'. These have been replaced with NaN.")
            else:
                raise KeyError(f"No mapping found for column '{col}' in label_encoders.")
        
        # Make prediction
        prediction = predictor.predict(input_df)
        
        return render_template('result.html', prediction=prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)