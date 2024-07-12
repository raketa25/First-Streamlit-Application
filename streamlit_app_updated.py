# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# import joblib
# import pickle

# # Load the dataset
# df = pd.read_csv("train.csv")

# # Remove irrelevant variables
# df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# # Create target variable and feature sets
# y = df['Survived']
# X_cat = df[['Pclass', 'Sex', 'Embarked']]
# X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

# # Handle missing values
# for col in X_cat.columns:
#     X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
# for col in X_num.columns:
#     X_num[col] = X_num[col].fillna(X_num[col].median())

# # Encode categorical variables
# X_cat_encoded = pd.get_dummies(X_cat, columns=X_cat.columns)

# # Concatenate the processed dataframes
# X = pd.concat([X_cat_encoded, X_num], axis=1)

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# # Standardize numerical values
# scaler = StandardScaler()
# X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
# X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# # Function to get scores
# def scores(clf, choice):
#     if choice == 'Accuracy':
#         return clf.score(X_test, y_test)
#     elif choice == 'Confusion matrix':
#         return confusion_matrix(y_test, clf.predict(X_test))

# # Three pages creation
# st.title("Titanic: Binary Classification Project")
# st.sidebar.title("Table of Contents")
# pages = ["Exploration", "DataVizualization", "Modelling"]
# page = st.sidebar.radio("Go to", pages)

# # Data Exploration
# if page == pages[0]:
#     st.write("### Presentation of Data")
#     st.dataframe(df.head(10))
#     st.write(df.shape)
#     st.dataframe(df.describe())

#     if st.checkbox("Show NA"):
#         st.dataframe(df.isna().sum())

# # DataVizualization
# if page == pages[1]:
#     st.write("### DataVizualization")

#     fig1 = plt.figure()
#     sns.countplot(x='Survived', data=df)
#     st.pyplot(fig1)

#     fig2 = plt.figure()
#     sns.countplot(x='Sex', data=df)
#     plt.title("Distribution of the Passengers Gender")
#     st.pyplot(fig2)

#     fig3 = plt.figure()
#     sns.countplot(x='Pclass', data=df)
#     plt.title("Distribution of the Passengers Class")
#     st.pyplot(fig3)

#     fig4 = sns.displot(x='Age', data=df)
#     plt.title("Distribution of the Passengers Age")
#     st.pyplot(fig4)

#     fig5 = plt.figure()
#     sns.countplot(x='Survived', hue='Sex', data=df)
#     st.pyplot(fig5)

#     fig6 = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
#     st.pyplot(fig6)

#     fig7 = sns.lmplot(x='Age', y='Survived', hue='Pclass', data=df)
#     st.pyplot(fig7)

#     fig8, ax = plt.subplots()
#     sns.heatmap(df.corr(), ax=ax)
#     st.pyplot(fig8)

# # Modelling
# if page == pages[2]:
#     st.write("### Modelling")

#     # Select model
#     choice = ['Random Forest', 'SVC', 'Logistic Regression']
#     option = st.selectbox('Choice of the model', choice)
#     st.write('The chosen model is:', option)

#     # Load model
#     if option == 'Random Forest':
#         clf = joblib.load("rf_model.joblib")
#     elif option == 'SVC':
#         clf = joblib.load("svc_model.joblib")
#     elif option == 'Logistic Regression':
#         clf = joblib.load("lr_model.joblib")

#     # Display model results
#     display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
#     if display == 'Accuracy':
#         st.write(scores(clf, display))
#     elif display == 'Confusion matrix':
#         st.dataframe(scores(clf, display))

########################## Final Version of the app ###########################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv("train.csv")

# Remove irrelevant variables
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Create target variable and feature sets
y = df['Survived']
X_cat = df[['Pclass', 'Sex', 'Embarked']]
X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

# Handle missing values
for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())

# Encode categorical variables
X_cat_encoded = pd.get_dummies(X_cat, columns=X_cat.columns)

# Concatenate the processed dataframes
X = pd.concat([X_cat_encoded, X_num], axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Standardize numerical values
scaler = StandardScaler()
X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# Function to get scores
def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))

# Three pages creation
st.title("Titanic: Binary Classification Project")
st.sidebar.title("Table of Contents")
pages = ["Exploration", "DataVizualization", "Modelling"]
page = st.sidebar.radio("Go to", pages)

# Data Exploration
if page == pages[0]:
    st.write("### Presentation of Data")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())

# DataVizualization
if page == pages[1]:
    st.write("### DataVizualization")

    fig1 = plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig1)

    fig2 = plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Distribution of the Passengers Gender")
    st.pyplot(fig2)

    fig3 = plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Distribution of the Passengers Class")
    st.pyplot(fig3)

    fig4 = sns.displot(x='Age', data=df)
    plt.title("Distribution of the Passengers Age")
    st.pyplot(fig4)

    fig5 = plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    st.pyplot(fig5)

    fig6 = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig6)

    fig7 = sns.lmplot(x='Age', y='Survived', hue='Pclass', data=df)
    st.pyplot(fig7)

    fig8, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.pyplot(fig8)

# Modelling
if page == pages[2]:
    st.write("### Modelling")

    # Select model
    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is:', option)

    # Load model
    if option == 'Random Forest':
        clf = joblib.load("rf_model.joblib")
    elif option == 'SVC':
        clf = joblib.load("svc_model.joblib")
    elif option == 'Logistic Regression':
        clf = joblib.load("lr_model.joblib")

    # Display model results
    display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))


# # streamlit run streamlit_app_updated.py