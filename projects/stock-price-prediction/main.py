# ==============================================================
# 📈 Stock Price Prediction using LSTM
# --------------------------------------------------------------
# This script downloads stock data from Yahoo Finance,
# trains an LSTM model to predict closing prices,
# plots actual vs predicted values, and saves both the plot
# and the trained model. Designed for GitHub Codespaces.
# ==============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 🔧 Suppress TensorFlow info & warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ✅ Ensure local src/ folder is on the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 🧩 Import helper functions from local modules
from utils import download_stock_data
from lstm_model import prepare_data, build_lstm_model

# ==============================================================
# 1️⃣ Load Data
# --------------------------------------------------------------
ticker = "AAPL"  # You can change this — e.g. "GOOG", "TSLA", "MSFT"

print(f"⬇️ Downloading stock data for {ticker}...")
data = download_stock_data(ticker)

# ==============================================================
# 2️⃣ Split Train/Test
# --------------------------------------------------------------
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

print(f"📊 Data split: {len(train_data)} train / {len(test_data)} test records")

# ==============================================================
# 3️⃣ Prepare Training Data
# --------------------------------------------------------------
X_train, y_train, scaler = prepare_data(train_data.values)
print(f"🧩 Training shape: {X_train.shape}, Labels: {y_train.shape}")

# ==============================================================
# 4️⃣ Build & Train the LSTM Model
# --------------------------------------------------------------
model = build_lstm_model((X_train.shape[1], 1))
print("🚀 Training the model — this may take a few seconds...")

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# ==============================================================
# 5️⃣ Prepare Test Data & Predict
# --------------------------------------------------------------
X_test, y_test, _ = prepare_data(test_data.values)
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Adjust true prices to match prediction length
true_prices = data['Close'].values[-len(predictions):]

# ==============================================================
# 6️⃣ Plot & Save Result
# --------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(true_prices, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title(f'{ticker} Stock Price Prediction with LSTM')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()

# Save the chart for Codespaces
plt.savefig('results.png')
print("✅ Chart saved as results.png (open it in the Codespaces file explorer)")

# ==============================================================
# 7️⃣ Save Trained Model
# --------------------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save('models/lstm_stock_model.h5')
print("✅ Model saved to models/lstm_stock_model.h5")

print("\n🎉 All done! You can now view results.png and your saved model.")