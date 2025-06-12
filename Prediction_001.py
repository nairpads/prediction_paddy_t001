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

    # ✅ Normalize column names to avoid case/space issues
    train.columns = train.columns.str.strip().str.lower()
    test.columns = test.columns.str.strip().str.lower()

    # Combine for preprocessing
    combine = [train, test]

    for dataset in combine:
        # Fill missing values
        if 'age' in dataset.columns:
            dataset['age'].fillna(dataset['age'].median(), inplace=True)
        if 'embarked' in dataset.columns:
            dataset['embarked'].fillna(dataset['embarked'].mode()[0], inplace=True)
        if 'fare' in dataset.columns:
            dataset['fare'].fillna(dataset['fare'].median(), inplace=True)

        # ✅ Map 'sex' safely, even if values aren't strings
        if 'sex' in dataset.columns:
            dataset['sex'] = dataset['sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
            dataset['sex'].fillna(-1, inplace=True)
            dataset['sex'] = dataset['sex'].astype(int)

        # ✅ Map 'embarked' safely
        if 'embarked' in dataset.columns:
            dataset['embarked'] = dataset['embarked'].astype(str).str.upper().apply(
                lambda x: {'S': 0, 'C': 1, 'Q': 2}.get(x, -1)
            ).astype(int)

    # ✅ Save PassengerId before dropping
    test_passenger_id = test['passengerid'].copy()

    # Drop unwanted columns
    drop_cols = ['name', 'ticket', 'cabin', 'passengerid']
    train = train.drop(columns=[col for col in drop_cols if col in train.columns])
    test = test.drop(columns=[col for col in drop_cols if col in test.columns])

    # Features and target
    X = train.drop("survived", axis=1)
    y = train["survived"]
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
    st.subheader("📊 Model Accuracy")
    st.write(f"🧮 Logistic Regression: **{acc_log:.4f}**")
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

else:
    st.info("👋 Please upload both training and test CSV files to get started.")
