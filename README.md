# House Price Prediction - Kaggle Competition

A comprehensive machine learning project for predicting house prices using advanced regression techniques, feature engineering, and ensemble methods.

## Project Overview

This project tackles the Kaggle House Prices competition, implementing a complete machine learning pipeline from data exploration to model deployment. The solution includes extensive feature engineering, multiple regression algorithms, and sophisticated ensemble methods to achieve optimal prediction accuracy.

## Project Structure

```
house-price-pred/
├── datasets/
│   ├── train.csv                    # Training dataset
│   ├── test.csv                     # Test dataset
│   ├── sample_submission.csv        # Sample submission format
│   └── submissions/                 # Model predictions
│       ├── xgboost_submission.csv
│       ├── rf_submission.csv
│       ├── mlp_submission.csv
│       ├── avg_ens_submission.csv
│       └── sub_stack.csv
├── docs/
│   └── data_description.txt         # Dataset feature descriptions
├── notebooks/
│   └── house_prices_pred.ipynb      # Main analysis notebook
└── README.md
```

## Analysis Pipeline

### 1. Data Exploration & Visualization
- **Dataset Overview**: 1460 training samples with 81 features
- **Missing Value Analysis**: Comprehensive handling of null values across features
- **Target Variable Analysis**: SalePrice distribution and normalization strategies
- **Feature Relationships**: Correlation analysis and visual exploration

### 2. Exploratory Data Analysis
Key insights explored:
- Building type distribution and price impact
- Zoning classification effects on pricing
- Street and alley access correlation with prices
- Living area vs. sale price relationships
- Property age impact on valuation
- Year-over-year pricing trends
- Lot shape and contour influences

### 3. Feature Engineering
Custom features created:
- `PropertyAge`: Calculated from year sold and year built
- `TotalSF`: Combined total square footage (basement + floors)
- `TotalBath`: Aggregated bathroom count
- `HasRemodeled`: Boolean flag for renovation history
- `HasSecondFloor`: Second floor presence indicator
- `HasGarage`: Garage availability flag
- Categorical encoding for temporal features

### 4. Data Preprocessing Pipeline
- **Numerical Features**: Missing value imputation (mean) + StandardScaler
- **Categorical Features**: Missing value imputation (constant) + OneHotEncoder
- **Dimensionality Reduction**: PCA with 95% variance retention
- **Target Transformation**: Log normalization for improved model performance

## Machine Learning Models

### Base Models Implemented
1. **Linear Regression**: Baseline model with regularization
2. **Random Forest**: Ensemble tree-based approach
3. **XGBoost**: Gradient boosting with hyperparameter tuning
4. **Multi-Layer Perceptron (MLP)**: Neural network regression

### Model Performance Comparison
Models evaluated using:
- 3-fold cross-validation
- Negative mean squared error scoring
- RMSE metrics on test data
- Grid search hyperparameter optimization

### Advanced Ensemble Methods
- **Average Ensemble**: Simple averaging of top-performing models
- **Stacking Ensemble**: Meta-learning approach with multiple base learners
- **Meta-Model Selection**: Automated best meta-model identification

## Results & Submissions

Multiple submission files generated:
- Individual model predictions (XGBoost, Random Forest, MLP)
- Average ensemble predictions
- Stacking ensemble predictions

## Technologies Used

- **Data Manipulation**: pandas, numpy
- **Visualization**: plotly (interactive charts)
- **Machine Learning**: scikit-learn, XGBoost
- **Statistical Analysis**: scipy.stats
- **Development Environment**: Jupyter Notebook

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost plotly scipy jupyter
```

### Running the Analysis
1. Clone the repository
2. Navigate to the project directory
3. Open `notebooks/house_prices_pred.ipynb` in Jupyter
4. Run all cells to reproduce the analysis

### Key Libraries Required
```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import xgboost as XGBRegressor
```

## Key Features

- **Comprehensive EDA**: 7+ visualization analyses covering all major data aspects
- **Robust Preprocessing**: Automated pipeline handling mixed data types
- **Advanced Feature Engineering**: Domain-specific feature creation
- **Model Comparison**: Systematic evaluation of multiple algorithms
- **Ensemble Learning**: Implementation of both simple and complex ensemble methods
- **Hyperparameter Optimization**: Grid search for optimal model parameters

## Competition Insights

- **Target Variable**: Log-transformed SalePrice for improved normalization
- **Feature Importance**: Living area, property age, and total square footage as key predictors
- **Missing Data Strategy**: Feature-specific imputation strategies
- **Validation Strategy**: 3-fold cross-validation for robust performance estimation

## Future Improvements

- Feature selection techniques (RFE, LASSO)
- Advanced ensemble methods (Bayesian optimization)
- Deep learning approaches
- Automated feature engineering
- Model interpretability analysis (SHAP values)

## License

This project is open source and available under the MIT License.

---

*This project demonstrates a complete machine learning workflow from data exploration to model deployment, showcasing best practices in data science and machine learning engineering.*
