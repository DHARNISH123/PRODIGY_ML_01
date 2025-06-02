import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'bathrooms': [2, 2, 3, 4, 3],
    'price': [300000, 350000, 400000, 500000, 450000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Show results
for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual}, Predicted: {predicted}")
