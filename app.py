import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
from sklearn.model_selection import train_test_split

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('ss.jpg')  

# Load the dataset into a pandas DataFrame
df = pd.read_csv('train.csv')

# Convert the date column to a datetime object
df['date'] = pd.to_datetime(df['date'])

# Create binary columns for each day of the week
day_of_week_columns = pd.get_dummies(df['date'].dt.day_name())
df = pd.concat([df, day_of_week_columns], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['store', 'item', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']], df['sales'], test_size=0.2, random_state=42)

# Train a decision tree regression model on the training set
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Calculate the accuracy of the model on the testing set
accuracy = model.score(X_test, y_test)
print("model used:", model)
print("r2:", accuracy)

# Define the input form
st.write("# Sales Prediction App")
date =st.selectbox(
    'Select a day',
    ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
store = st.number_input("Enter the store number", min_value=1, max_value=10)
item = st.number_input("Enter the item number", min_value=1, max_value=50)

# Convert the input date to a binary column
day_column = pd.DataFrame(columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
day_column[date] = 1

# Make a prediction using the input values
input_values = pd.concat([pd.DataFrame({'store': [store], 'item': [item]}), day_column], axis=1)
prediction = model.predict(input_values)

# Display the prediction to the user
if st.button("Predict"):
    st.write("The predicted sales for {} at store {} for item {} is {}".format(date, store, item, prediction))


