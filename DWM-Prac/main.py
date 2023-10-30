import pickle
import streamlit as st

model = pickle.load(open("trained.pkl", "rb"))

st.title("House Price Prediction")

# Take user input for the features
area = st.number_input("Enter the area", min_value=5000, max_value=10000)
bedrooms = st.number_input("Enter the number of bedrooms", min_value=1, max_value=5)
mainroad = st.checkbox("Is it on the main road?")
guestroom = st.checkbox("Does it have a guest room?")
basement = st.checkbox("Does it have a basement?")
parking = st.number_input("Enter the number of parking spaces", min_value=1, max_value=2)
furnishingstatus = st.radio("Select furnishing status", ["Furnished", "Semifurnished", "Unfurnished"])

if furnishingstatus == "Furnished":
    furnishingstatus = 1
elif furnishingstatus == "Semifurnished":
    furnishingstatus = 2
else:
    furnishingstatus = 0

if mainroad == True:
    mainroad = 1
else:
    mainroad = 0

if guestroom == True:
    guestroom = 1
else:
    guestroom = 0

if basement == True:
    basement = 1
else:
    basement = 0


# Add a submit button
if st.button("Predict Price"):
    # Make prediction
    features = [area, bedrooms, mainroad, guestroom, basement, parking, furnishingstatus]
    predicted_price = model.predict([features])[0]

    # Display the predicted price
    st.success(f"Predicted Price: ₹{predicted_price}")
