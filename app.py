from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/funeral_insurance_data_model_20230611194858.joblib')
sc = joblib.load('models/funeral_insurance_data_scaler_20230611194858.joblib')

# Function for preprocessing input values and performing necessary encoding and feature engineering
def preprocess_input(input_data):
    # Create a DataFrame from the input data
    # input_df = pd.DataFrame(input_data, columns=['Age', 'Gender', 'Education Level', 'Marital Status', 'Income'])
	
    print(input_data)
    input_df = pd.DataFrame([input_data])
    

    input_df['Married'] = input_df['Married'].apply(lambda x: 1 if x == 'Married' else 0)
    gender_mapping = {'Male': 0, 'Female': 1}
    input_df['Gender'] = input_df['Gender'].map(gender_mapping)
    education_categories = {
        'Primary School': 'Low',
        'High School': 'Medium',
        'College': 'High',
        'Bachelors Degree': 'High',
        'Masters Degree': 'High',
        'PhD': 'High'
    }
    # Create a new column 'EducationLevelCategory' to store the categorized education level
    input_df['Education Level'] = input_df['Education Level'].map(education_categories)

    income_thresholds = [0, 50000, 100000, float('inf')]  # Define income thresholds

    # Create a new column 'IncomeLevel' to store the categorized income level
    input_df['IncomeLevel'] = pd.cut(input_df['IncomeLevel'], bins=income_thresholds, labels=['Low', 'Medium', 'High'], right=False)


    # Define the categorical columns to be one-hot encoded
    categorical_columns = ['IncomeLevel', 'Education Level']

    # Perform one-hot encoding for the categorical columns
    encoded_features = pd.get_dummies(input_df[categorical_columns])

    # Merge the one-hot encoded features with the original dataframe
    input_df_final = pd.concat([input_df, encoded_features], axis=1)

    # Drop the original categorical columns after encoding
    input_df_final.drop(categorical_columns, axis=1, inplace=True)
    

    # Read the column names from the file
    column_names_df = pd.read_csv('models/encoded_column_names_20230612172053.csv')

    # Get the column names as a list
    saved_column_names = column_names_df['Column Names'].values.tolist()

    # Get the missing columns
    missing_columns = list(set(saved_column_names) - set(input_df_final.columns))

    # Add the missing columns to the input DataFrame with zeros as values
    for column in missing_columns:
        if column != 'PolicyUptake':
            input_df_final[column] = 0


    input_df_final = sc.transform(input_df_final)

    # Reshape data for input into our model predict function
    input_df_final = input_df_final.reshape(1, -1)

    print(input_df_final)
    # Perform necessary preprocessing steps, such as encoding and feature engineering
    # ...

    return input_df_final

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input values from the form
        age = int(request.form['age'])
        gender = request.form['gender']
        education = request.form['education_level']
        married = request.form['marital_status']
        income = float(request.form['income'])

        # Create a dictionary with the input values
        input_data = {'Age': age, 'Gender': gender, 'Education Level': education, 'Married': married, 'IncomeLevel': income}

        # Preprocess the input values
        input_df = preprocess_input(input_data)

        # Make a prediction using the preprocessed input
        prediction = model.predict(input_df)

        # Convert the prediction to a human-readable label
        prediction_label = 'Yes' if prediction == 1 else 'No'

        return render_template('index.html', prediction=prediction_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
