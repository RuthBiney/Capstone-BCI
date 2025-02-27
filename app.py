import streamlit as st
import requests
import numpy as np

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"  # Update this if running on a different server

st.title("Movement Classification App")

# Input field for movement keypoints
st.write("Enter keypoints data as a comma-separated list (e.g., x1, y1, x2, y2, ...):")
keypoints_input = st.text_area("Keypoints Data", "")

if st.button("Predict Movement"):
    try:
        # Convert input to list of floats
        keypoints_list = np.array([float(i) for i in keypoints_input.split(",")], dtype=np.float32).tolist()

        # Send request to Flask API
        response = requests.post(API_URL, json={"keypoints": keypoints_list})
        result = response.json()

        # Display prediction
        if "prediction" in result:
            st.success(f"Predicted Movement Class: {result['prediction']}")
        else:
            st.error("Error: Could not get prediction.")

    except Exception as e:
        st.error(f"Invalid input: {e}")
