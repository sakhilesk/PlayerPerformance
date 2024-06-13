# Player Performance Analysis

## Introduction
This project aims to analyze and compare the performance of soccer players using historical data. The analysis includes data cleaning, feature extraction, statistical analysis, visualizations, a player ranking system, and predictions using machine learning models.

## Project Structure
The project is structured as follows:
- `data/`: Contains the datasets used for analysis.
- `notebooks/`: Jupyter notebooks with the step-by-step analysis.
- `src/`: Source code for data processing, feature extraction, and modeling.
- `visualizations/`: Visualizations generated during the analysis.
- `README.md`: Project overview and instructions.

## Datasets
The datasets used in this project can be found in the `data/` directory. You can replace these with any relevant player performance datasets from sources like FIFA, OPTA, or publicly available player stats.

## Requirements
To run the project, you need the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using:
```bash
pip install -r requirements.txt

Analysis Steps
1. Data Collection
Load the dataset using pandas:

import pandas as pd

# Load the dataset
data = pd.read_csv('data/player_stats.csv')

2. Data Cleaning and Preprocessing
Clean and preprocess the data for analysis:

# Check for missing values
data.isnull().sum()

# Fill or drop missing values
data.fillna(0, inplace=True)

3. Feature Extraction
Extract relevant features such as goals, assists, passes, etc.:

features = ['goals', 'assists', 'passes', 'shots', 'minutes_played']
player_data = data[features]

4. Statistical Analysis and Visualization
Perform statistical analysis and create visualizations:

import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of goals
plt.figure(figsize=(10, 6))
sns.histplot(player_data['goals'], bins=20, kde=True)
plt.title('Distribution of Goals')
plt.xlabel('Goals')
plt.ylabel('Frequency')
plt.show()

5. Player Ranking System
Build a player ranking system based on performance:

# Calculate a simple ranking score
player_data['performance_score'] = (player_data['goals'] * 4 + 
                                    player_data['assists'] * 3 + 
                                    player_data['passes'] * 0.1)

# Rank players based on performance score
player_data['rank'] = player_data['performance_score'].rank(ascending=False)

# Display top 10 players
top_10_players = player_data.sort_values(by='performance_score', ascending=False).head(10)
print(top_10_players)

6. Predict Player Performance
Use machine learning to predict future player performance:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target
X = player_data[['assists', 'passes', 'shots', 'minutes_played']]
y = player_data['goals']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Goals')
plt.ylabel('Predicted Goals')
plt.title('Actual vs Predicted Goals')
plt.show()

Conclusion
This project provides a comprehensive analysis of soccer player performance, including data cleaning, feature extraction, statistical analysis, visualizations, a player ranking system, and predictions using machine learning models. You can expand this project by adding more sophisticated models, feature engineering, or incorporating additional data sources for a more comprehensive analysis.

Author
Sakhile harrison Sithole
