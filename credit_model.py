import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_engine import create_credit_data

FEATURES = [
    'Beneish_M','Altman_Z','Sloan_Ratio',
    'Piotroski_F','DSR','Current_Ratio',
    'Working_Cap_Days','Auditor_Qualification'
]

@st.cache_resource
def load_model():
    data = create_credit_data()
    X, y = data[FEATURES], data['fraud_flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)
    accuracy = (model.predict(X_test) == y_test).mean()

    return model, accuracy, data
