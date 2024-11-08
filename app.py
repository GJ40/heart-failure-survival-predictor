# import required libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Load the trained model and training data
model = joblib.load('trained_model/logistic_regression_model.joblib')
training_data = pd.read_csv('dataset/heart_failure_clinical_records_dataset.csv') 

# Function for scaling input data based on mean and std deviation of training data
def scale_input(data, training_data):
    return (data - training_data.mean()) / training_data.std()

# prediction function
def predict_outcome(input_data_scaled):
    prediction = model.predict(input_data_scaled)
    return "Death Event" if prediction[0] == 1 else "Survival"


# function to handle missing values
def preprocess_input_data(data):
    imputer = SimpleImputer(strategy="mean")
    data_imputed = imputer.fit_transform(data)
    return data_imputed


# tabs
tab1, tab2 = st.tabs(['Home', 'Charts'])
with tab1:
    st.title(":heart: Heart Failure Prediction")
    st.markdown("### Enter patient details to predict heart failure outcome.")

    # Collect user input for each feature
    age = st.slider("Age", min_value=0, max_value=120, value=18)
    sex = st.radio("Sex", options=['female', 'male'], index=None)
    anaemia = st.checkbox("Does patient have anaemia", value=False)
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=90)
    diabetes = st.checkbox("Does patient have diabetes", value=False)
    ejection_fraction = st.slider("Ejection Fraction (%)", min_value=0, max_value=100, value=55)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=0.9)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0, value=138)
    time = st.number_input("Time (follow-up period in days)", min_value=0, value=0)

    # Organize user input into a DataFrame
    new_data = pd.DataFrame({
        'age': [age],
        'anaemia': [1 if anaemia else 0],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [0 if sex=='female' else 1],
        'time': [time]
    })

    # Scale the new input data using training data
    new_data_scaled = scale_input(new_data, training_data.drop(columns='DEATH_EVENT'))

    # Predict outcome when the button is pressed
    if st.button("Predict"):
        # Preprocess the scaled data
        #print(age, anaemia,creatinine_phosphokinase, diabetes, ejection_fraction, serum_creatinine, serum_sodium, sex, time)
        new_data_scaled_imputed = preprocess_input_data(new_data_scaled)
        result = predict_outcome(new_data_scaled_imputed)
        emoji = ':broken_heart:' if result=='Death Event' else ':heartbeat:'
        st.markdown(f'# Prediction: {emoji} {result}')

        # Add new data point training data
        combined_data = training_data.copy()
        combined_data['New Data'] = "Existing"
        new_data_visual = new_data.copy()
        new_data_visual['DEATH_EVENT'] = 1 if result == "Death Event" else 0
        new_data_visual['New Data'] = "New Data"
        combined_data = pd.concat([combined_data, new_data_visual])

        st.markdown('---')

        # Visualize data 
        st.write("Position of the new data point among the training data:")
        plt.figure(figsize=(10, 6))
        st.scatter_chart(
            data=combined_data,
            x="ejection_fraction",
            x_label="ejection_fraction %" ,
            y="serum_creatinine",
            y_label="serum_creatinine mg/dL",
            color="New Data"
        )

        _ = '''
        sns.scatterplot(
            data=combined_data,
            x="ejection_fraction",
            y="serum_creatinine",
            hue="New Data",
            style="DEATH_EVENT",
            palette={"Existing": "blue", "New Data": "red"},
            markers={0: "o", 1: "X"}
        )
        
        plt.title("Comparison of New Data with Training Data")
        plt.xlabel("Ejection Fraction (%)")
        plt.ylabel("Serum Creatinine (mg/dL)")
        st.pyplot(plt.gcf())
        '''

#with tab2:
#    pass
