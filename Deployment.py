#-*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""



import pandas as pd
import streamlit as st 
from sklearn.ensemble import RandomForestRegressor

st.title('Model Deployment:RandomForestRegression')

def user_input_features():
        Temperature=st.text_input('Temperature')
        Exhaust_vacuum=st.text_input('Exhaust_vacuum')
        Ambient_pressure=st.text_input('Ambient_pressure')
        Relative_Humidity=st.text_input('Relative_Humidity')
        Energy_production=st.text_input('Energy_production')
        data = {'Temperature':Temperature,
            'Exhaust_vacuum':Exhaust_vacuum,
            'Ambient_pressure':Ambient_pressure,
            'Relative_Humidity':Relative_Humidity,
            'Energy_production':Energy_production}

        features = pd.DataFrame(data,index = [0])
        return features 


df = user_input_features()
df = df.rename(columns={"Ambient_pressure": "amb_pressure", "Exhaust_vacuum": "exhaust_vacuum", "Relative_Humidity": "r_humidity", "Temperature": "temperature"})
st.subheader('User Input parameters')
st.write(df)


data = pd.read_csv('Regrerssion_energy_production_data.csv',sep=';')

X = data.iloc[:,0:4]
y = data["energy_production"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


random_forest = RandomForestRegressor(n_estimators=100, random_state=42)  
random_forest.fit(X_train, y_train)


predictions = random_forest.predict(X_test)
st.subheader(f'Prediction Result: {predictions}')
st.write(predictions)

params = [
    {'n_estimators': 50, 'max_depth': 5, 'random_state': 42},
    {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
    {'n_estimators': 150, 'max_depth': None, 'random_state': 42},
]

results=[]

for i, param_set in enumerate(params):

    random_forest = RandomForestRegressor(**param_set)
    print(f"Training with parameters: {param_set}")
    
try:
    random_forest.fit(X_train, y_train)
    prediction = random_forest.predict(X_test)
    results.append({'Parameters': param_set, 'Prediction': prediction})

except Exception as e:
        print(f"Error during training: {e}")
result_df = pd.DataFrame(results)



st.subheader('Prediction Results for Different Parameter Sets')
st.write(result_df)







