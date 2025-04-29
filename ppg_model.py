import joblib
import pandas as pd


model = joblib.load('HR_xgb2.pkl')
scaler = joblib.load('scaler2.pkl')

def data_process( age, hr):
    sample = {'Age': [age], 'HR': [hr]}  

    data = pd.DataFrame(sample)
    # data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    # Create the new parameters
    data['age_hr_interaction'] = data['Age'] * data['HR']
    data['age_normalized_by_hr'] = data['Age'] / (data['HR'] + 1)

    # Step 3: Transform the single sample
    data[['Age', 'HR', 'age_hr_interaction', 'age_normalized_by_hr']] = scaler.transform(
        data[['Age', 'HR', 'age_hr_interaction', 'age_normalized_by_hr']])

    # Now data is ready to be used as input to your model
    # print(data)
    return(data)

def model_pred(data):
    y_pred = model.predict(data)
    if y_pred[0] == 0:
        result = 'Normal'
    else:
        result = 'Abnormal'

    print(f"Prediction: {result}")

data = data_process(25,85)
model_pred(data)


