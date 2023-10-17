import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_model():
    with open('model.pkl', 'rb') as file:
        model_deploy2 = pickle.load(file)
    return model_deploy2


model_deploy2 = load_model()

def show_predict_page():
    st.title("Software Developer Salary Predition Model")

    st.header('Enter Your Information:')
    years_code_pro = st.slider('Years of Professional Coding Experience', 0, 50, 5)  # Slider for numerical input
    dev_type = st.selectbox('Developer Type', ["Software Developer",
                                               "Data Scientist/Engineer",
                                               'IT role',
                                               'Developer, mobile',
                                               "Other",
                                               "Academia",
                                               "Senior Executive",
                                               "Non-technical ",
                                               "Academic researcher",
                                               "Developer Advocate",
                                               "Database administrator"
                                               ])  # Dropdown for categorical input
    ed_level = st.selectbox('Education Level', ["Less than a Bachelors'",
                                                "Bachelor’s degree",
                                                "Master’s degree",
                                                "Professional degree"])  # Dropdown for categorical input
    ic_or_pm = st.radio('Individual Contributor or People Manager',
                        ['Individual contributor', 'People manager'])  # Radio button for categorical input
    country = st.selectbox('Country', ["Australia",
                                       "Brazil",
                                       "Canada",
                                       "Denmark",
                                       "France",
                                       "Germany",
                                       "India",
                                       "Israel",
                                       "Italy",
                                       "Netherlands",
                                       "Poland",
                                       "Portugal",
                                       "Spain",
                                       "Sweden",
                                       "Switzerland",
                                       "UK & N.I.",
                                       "USA",
                                       "Other"
                                       ])
    if st.button('Predict Salary'):
        input_data = pd.DataFrame({
            'YearsCodePro': [years_code_pro],
            'DevType': [dev_type],
            'EdLevel': [ed_level],
            'ICorPM': [ic_or_pm],
            'Country': [country],
        })

        # Predict using the linear regression model
        predicted_salary = model_deploy2.predict(input_data)

        st.subheader('Predicted Salary:')
        st.write(f'${predicted_salary[0]:,.2f}')


show_predict_page()
