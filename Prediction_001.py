import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🚢 Titanic Survival Prediction")
st.write("Upload your **train.csv** and **test.csv** files to begin.")

# File upload
train_file = st.file_uploader("Upload train.csv", type=["csv"])
test_file = st.file_uploader("Upload test.csv", type=["csv"])

if train_file is not None and test_file is not None:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    # Combine for preprocessing
    combine = [train, test]

    for dataset in combine:
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Save PassengerId for submission
    test_passenger_id = test['PassengerId']

    # Drop unwanted columns
    train = train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    test = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # Features and target
    X = train.drop("Survived", axis=1)
    y = train["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_val)
    acc_log = accuracy_score(y_val, y_pred_log)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)

    # Show accuracy
    st.subheader("Model Accuracy")
    st.write(f"📊 Logistic Regression: **{acc_log:.4f}**")
    st.write(f"🌲 Random Forest: **{acc_rf:.4f}**")

    # Final prediction and submission
    final_predictions = rf.predict(test)
    submission = pd.DataFrame({
        "PassengerId": test_passenger_id,
        "Survived": final_predictions
    })

    # Provide download
    st.subheader("📥 Download Submission File")
    csv = submission.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download submission.csv",
        data=csv,
        file_name='submission.csv',
        mime='text/csv'
    )
