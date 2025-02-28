import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/training_data.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Normalize dataset
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Save processed data
df_scaled.to_csv("data/processed_data.csv", index=False)
print("✅ Preprocessed data saved.")
