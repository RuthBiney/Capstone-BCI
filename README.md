**Hybrid CNN-RNN for Movement Classification**
**Overview**
This project develops a movement classification system using a hybrid CNN-RNN model. The model is trained on pose keypoints extracted from movement videos, enabling real-time classification of movement patterns. The system is designed for applications in sports analytics, physiotherapy, and motion-based human-computer interaction.

    The project consists of:
    ✅ A CNN-RNN deep learning model for movement classification.
    ✅ A Flask API for real-time movement prediction.
    ✅ A Streamlit web app for an interactive user interface.
    ✅ A fully documented workflow for training, deployment, and testing.

**Features**

- Hybrid CNN-RNN Model: Uses CNNs for spatial feature extraction and RNNs (LSTM) for temporal dependencies in movement data.
- Pose Keypoint-Based Movement Analysis: Classifies movements using extracted pose keypoints from video datasets.
- Real-Time Classification: Predicts movements with low latency via a Flask API.
- User-Friendly Interface: A Streamlit-based UI for easy interaction and visualization.
- Modular & Open-Source: Fully documented and easy to extend for further research.

**Installation**

1.  **_Clone the Repository_**
    https://github.com/RuthBiney/Capstone-BCI.git
2.  **_Set Up a Virtual Environment._**
    python -m venv venv

- **_Activate the virtual environment._**
  - Windows : venv\Scripts\Activate.ps1
  - Mac/Linux : source venv/bin/activate

3. **_Install Dependencies._**
   pip install -r requirements.txt
4. **_Download the dataset_**

**USAGE**

1. **_Run all the script in the colab which,_**

- Normalizes the movement data.
- Handles missing values.
- Segments data into time windows for training.
- Loads the preprocessed movement data.
- Defines and compiles the CNN-RNN hybrid model.

2. **_Start the Flask API for predictions_**
   python flask_api.py
   If Flask is running successfully, you should see;

- Running on http://127.0.0.1:5000

3. **_Test API_**

- Option 1: Use Postman
  - Send a POST request to http://127.0.0.1:5000/predict
  - Body (JSON Format):
    {"keypoints": [0.5, 0.2, 0.6, 0.3, 0.7, 0.4, 0.8, 0.5, 0.9, 0.6]}.
  - Expected Response
    {"prediction": 1}
- Option 2: Use a Python Script (test_api.py)
  - Run the script:
    python test_api.py
  - Expected output:
    Response: {"prediction": 1}

**Running the Streamlit Web App**

1. Install Streamlit (If Not Installed)

- pip install streamlit requests numpy

2. Start the Web App

- streamlit run streamlit_app.py
  This will open the UI at: http://localhost:8501
  Users can input pose keypoints and see real-time predictions.

**Model Architecture**
The CNN-RNN hybrid model consists of:

1. CNN Layers (Spatial Feature Extraction)

- Conv1D (32 filters, kernel=3, ReLU activation)
- MaxPooling1D (2x2 window)
- Conv1D (64 filters, kernel=3, ReLU activation)
- MaxPooling1D (2x2 window)

2. RNN Layers (Temporal Feature Extraction)

- LSTM (128 units, return sequences=True)
- Dropout (0.5 for regularization)
- LSTM (64 units, return sequences=False)

3. Fully Connected Layer

- Dense(2, softmax activation) → Predicts movement class (e.g., 0 or 1).

**Testing with Local Users**

- The system does not require an EEG device for testing.
- Users can input pose keypoints manually or use automated video processing.
- Future iterations may incorporate EEG data for mind-controlled movement classification.

**Deployment**

1. Local Deployment
   Run the Streamlit app on your local machine:

- streamlit run streamlit_app.py

2. Cloud Deployment (Streamlit Sharing or Heroku)
   Option 1: Deploy on Streamlit Cloud

- Push the project to GitHub.
- Connect the repository to Streamlit Sharing.
  Option 2: Deploy on Heroku
  heroku create movement-classifier-app
  git push heroku main
  heroku open

**Screenshots**
Flask API Running

Streamlit Web App

**Contributors**
Your Name - Ruth Senior Biney
GitHub - https://github.com/RuthBiney
Contact: r.biney@alustudent.com
