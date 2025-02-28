import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer  # ✅ For handling missing values

# ✅ Step 1: Load dataset
df = pd.read_csv("data/training_data.csv")

# ✅ Step 2: Handle Missing Values
imputer = SimpleImputer(strategy="mean")  # Replace NaNs with column mean
X = imputer.fit_transform(df.values)

# ✅ Step 3: Check for remaining NaNs
if np.isnan(X).sum() > 0:
    print("Warning: NaNs still present after imputation! Filling with 0s.")
    X = np.nan_to_num(X)  # Replace any remaining NaNs with 0

# ✅ Step 4: Normalize Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ Step 5: Generate Labels Using K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y = kmeans.fit_predict(X)

# ✅ Step 6: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ✅ Step 7: Define the CNN-RNN Model
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
        self.fc = nn.Linear(128, 2)  # 2 output classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Reshape for RNN (batch, sequence, features)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ✅ Step 8: Train the Model
model = MovementClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ✅ Step 9: Save the Model
torch.save(model.state_dict(), "movement pth/best_movement_classifier.pth")
print("✅ Model saved as best_movement_classifier.pth")
