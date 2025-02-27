from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

# Define the exact same CNN-RNN model used during training
class MovementClassifier(nn.Module):
    def __init__(self):
        super(MovementClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 2)  # 2 output classes (same as training)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Reshape for RNN (batch, sequence, features)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Load the trained model
model = MovementClassifier()
model.load_state_dict(torch.load("best_movement_classifier.pth"))
model.eval()

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        keypoints = np.array(data["keypoints"]).astype(np.float32)  # Convert input to NumPy array
        
        # Convert to PyTorch tensor
        keypoints_tensor = torch.tensor(keypoints).unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        with torch.no_grad():
            output = model(keypoints_tensor)
            _, predicted_class = torch.max(output, 1)

        return jsonify({"prediction": int(predicted_class.item())})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
