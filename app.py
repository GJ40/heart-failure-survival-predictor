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
tab1, tab2, tab3 = st.tabs(['Home', 'Charts', 'About'])
with tab1:
    st.title(":heart: Heart Failure Survial Predictor")
    st.markdown("#### Enter patient details to predict heart failure outcome.")

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
        st.scatter_chart(
            data=combined_data,
            x="ejection_fraction",
            x_label="ejection_fraction %" ,
            y="serum_creatinine",
            y_label="serum_creatinine mg/dL",
            color="New Data"
        )
        _ = """
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
<<<<<<< HEAD
        """

with tab2:
    from math import ceil

    page_size = 10
    page_number = st.slider(
        label="Page Number",
        min_value=1,
        max_value=ceil(len(training_data)/page_size),
        step=1,
    )
    current_start = (page_number-1)*page_size
    current_end = page_number*page_size
    st.markdown("#### Dataset")
    st.write(training_data[current_start:current_end])

    st.markdown("---")

with tab3:
    st.markdown("""
    ### Why there are only 9 parameters for prediction?

    ---

    ##### Answer
    The decision to limit the feature set is likely driven by both medical relevance 
    and statistical considerations to improve model performance.

    ##### Medical and Statistical Basis for Feature Selection

    """)
    
    with st.expander("Improved Model Accuracy and Performance"):
        st.markdown("""
        Feature selection helps to enhance the predictive power of machine learning models by focusing only on the most relevant variables. Including too many features, especially those that are not strongly correlated with the target outcome (like heart failure prediction), can introduce noise, overfitting, and increase computational complexity. Studies have shown that using feature selection methods such as statistical tests (t-tests, Chi-squared tests) can significantly improve prediction accuracy for clinical models by focusing on high-impact factors like age, blood pressure, and specific lab results (e.g., serum creatinine levels) rather than a broad, less targeted set of features​
        """)
    
    with st.expander("Clinical Significance"):
        st.markdown("""
        When dealing with heart failure data, certain clinical indicators have a stronger association with patient outcomes. For instance:

        - Age and serum creatinine are critical factors for assessing kidney function, which is a significant risk factor for heart failure.
        - Ejection Fraction is a measure of how well the heart is pumping and is directly linked to heart health.
        - Serum Sodium levels can indicate fluid retention, which is crucial in heart failure cases. Reducing the number of features to only the most medically relevant ones, such as these, helps focus the model on clinically significant predictors, leading to better interpretability and trust in the results from a healthcare perspective​

        """)
    
    with st.expander("Benefits of Reducing Feature Count"):
        st.markdown("""
        - Computational Efficiency: Fewer features mean less computation, which is beneficial for deploying models in real-time applications like a Streamlit app.
        - Improved Data Quality: It reduces the chances of missing values (NaNs) affecting the model, as seen in your error logs, where the model encountered NaN values not handled by the Logistic Regression algorithm.
        - Easier Interpretation: Clinicians prefer models that are easier to interpret, especially for critical decisions. By using only the most impactful features, the model's decisions align better with medical understanding, making it more acceptable in a clinical setting​

        """)
    
    st.markdown("""
    ##### For more information you can visit:

    - <a href="https://ieeexplore.ieee.org/document/10180853">IEEE Paper on Enhanced Heart Failure Prediction Using Feature Selection.</a>
    
    - <a href="https://journals.plos.org/">Feature Selection Techniques for Heart Failure Prediction by PLOS ONE.</a>

    """, unsafe_allow_html=True)
