import pandas as pd

# Load the dataset (replace 'path_to_dataset' with the actual path)
data = pd.read_csv('Player States.csv')

# Step 2: Data Cleaning and Preprocessing
# Display the first few rows of the dataset
print(data.head())

# check for missing values
print(data.isnull().sum())

# fill or drop missing values as needed
data.fillna(0, inplace=True)

# Step 3: Feature Extraction
# Extract relevant features
features = ['Player_name','Goals','Assists','Passes','Shots','Minutes_Played']
player_data = data[features]

# Step 4: Statistical Analysis and Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of goals
plt.figure(figsize=(10, 6))
sns.histplot(player_data['Goals'], bins=20, kde=True)
plt.title('Distribution of goals')
plt.xlabel('Players')
plt.ylabel('Frequency')
plt.show()

# Step 5: Player Ranking System
# Calculate a simple ranking score

player_data['performance_score'] =(
    player_data['Goals'] * 4 +
    player_data['Assists'] * 3 +
    player_data['Passes'] * 0.1)

# Rank players based on performance score
player_data['rank'] = player_data['performance_score'].rank(ascending=False)

# Display top 10 players
top_10_players = player_data.sort_values(by='performance_score',ascending=False).head(10)
print(top_10_players)

# Step 6: Predict Player Performance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define features and target
x = player_data[['Assists','Passes','Shots','Minutes_Played']]
y = player_data['Goals']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Goals')
plt.ylabel('Predicted Goals')
plt.title('Actual vs Predicted Goals')
plt.show()

