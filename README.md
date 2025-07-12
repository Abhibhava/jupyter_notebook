# Linear Regression on California Housing Dataset

## What is Linear Regression?

Linear Regression is a **supervised learning algorithm** that predicts a continuous target variable by finding the best linear relationship between input features and the target.

### Key Concepts

**The Linear Equation:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
- `y` = target variable (house price)
- `β₀` = intercept (bias term)
- `β₁, β₂, ..., βₙ` = coefficients (weights) for each feature
- `x₁, x₂, ..., xₙ` = input features
- `ε` = error term

**How it Works:**
1. **Training**: Algorithm finds the best line/hyperplane that minimizes prediction errors
2. **Prediction**: Uses the learned equation to predict new values
3. **Evaluation**: Measures how well predictions match actual values

**Key Assumptions:**
- **Linearity**: Relationship between features and target is linear
- **Independence**: Observations are independent of each other
- **Homoscedasticity**: Constant variance in residuals
- **Normality**: Residuals are normally distributed

**Advantages:**
- Simple and interpretable
- Fast training and prediction
- No hyperparameter tuning required
- Good baseline for comparison
- Provides feature importance through coefficients

**Disadvantages:**
- Assumes linear relationship
- Sensitive to outliers
- Poor performance with non-linear data
- Requires feature scaling for optimal performance

**When to Use:**
- Regression problems (predicting continuous values)
- When you need interpretable results
- As a baseline model
- When relationship appears linear
- Small to medium-sized datasets

## Project Overview
This project demonstrates the implementation of **Linear Regression** on the California Housing dataset to predict house prices. The goal is to understand how linear regression works in a supervised learning context and evaluate its performance on real-world data.

## Dataset
- **Source**: Scikit-learn's California Housing dataset
- **Target**: Median house value (in hundreds of thousands of dollars)
- **Features**: 8 numerical features including median income, house age, average rooms, etc.
- **Size**: 20,640 samples

### Features Description
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group  
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

## Project Structure
```
linear_regression_project/
├── main.ipynb              # Main implementation file
├── README.md           # This file
└── .gitignore    # Dependencies (if needed)
```

## Implementation Steps

### 1. Data Loading and Preparation
- Load the California Housing dataset using `fetch_california_housing()`
- Convert sklearn dataset to pandas DataFrame for easier manipulation
- Add target variable as a new column named 'Target'

### 2. Data Splitting
- Separate features (X) from target variable (Y)
- Split data into training (80%) and testing (20%) sets
- Use `random_state=42` for reproducible results

### 3. Model Training
- Initialize Linear Regression model
- Train the model on training data (`X_train`, `Y_train`)
- Generate predictions on test data (`X_test`)

### 4. Model Evaluation
- Calculate Mean Squared Error (MSE) to measure prediction accuracy
- Calculate R² Score to measure model's explanatory power
- Visualize actual vs predicted values with error-coded scatter plot

## Key Results
- **Mean Squared Error**: ~0.58 (equivalent to ~$76,000 average error)
- **R² Score**: Measures how well the model explains the variance in house prices
- **Visualization**: Color-coded scatter plot showing prediction errors

## Code Highlights

### Data Preparation
```python
# Load and convert to DataFrame
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
```

### Model Implementation
```python
# Split data
X = df.drop('Target', axis=1)
Y = df['Target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
```

### Visualization
```python
# Error-coded scatter plot
errors = np.abs(Y_test - y_pred)
scatter = plt.scatter(Y_test, y_pred, c=errors, cmap='viridis', alpha=0.6, s=30)
plt.colorbar(scatter, label='Prediction Error')
```

## Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Installation and Usage
1. Install required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Learning Outcomes
- Understanding of supervised learning workflow
- Implementation of linear regression from scratch using scikit-learn
- Data preprocessing and train-test split methodology
- Model evaluation using MSE and R² metrics
- Data visualization techniques for model performance analysis

## Model Performance Analysis
- **MSE of 0.58**: Indicates average prediction error of ~$76,000
- **Relative Error**: ~38% on average house price of $200,000
- **Performance**: Reasonably good for real estate prediction, with room for improvement

### Understanding the Metrics

**Mean Squared Error (MSE):**
- Measures average of squared differences between actual and predicted values
- Lower values = better performance
- Units: squared units of target variable
- Formula: `MSE = (1/n) * Σ(actual - predicted)²`

**R² Score (Coefficient of Determination):**
- Measures proportion of variance in target explained by the model
- Range: 0 to 1 (higher is better)
- R² = 0.7 means model explains 70% of variance
- Formula: `R² = 1 - (SS_res / SS_tot)`

## Real-World Application
This model could be used by:
- **Real Estate Agents**: Quick price estimates
- **Property Investors**: Investment decisions
- **Banks**: Mortgage lending decisions
- **Government**: Housing policy planning

## Potential Improvements
- Feature engineering (polynomial features, interaction terms)
- Regularization techniques (Ridge, Lasso regression)
- Feature scaling and normalization
- Cross-validation for more robust evaluation
- Trying different algorithms (Random Forest, XGBoost)

## Visualization Features
- **Color-coded scatter plot**: Shows prediction errors visually
- **Perfect prediction line**: Red dashed line showing ideal predictions
- **Error colorbar**: Darker colors indicate higher prediction errors
- **Grid and transparency**: Enhances plot readability



## Contributing
Feel free to fork this repository and submit pull requests for improvements!

## License
This project is for educational purposes - feel free to use and modify for learning.
