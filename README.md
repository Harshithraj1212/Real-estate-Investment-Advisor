# Real Estate Investment Recommendation System

A machine learning project that predicts good real estate investment opportunities and forecasts future property prices using classification and regression models.

## Overview

This project analyzes Indian real estate data to help investors make informed decisions by:
- **Classifying** properties as good or poor investments
- **Predicting** future property prices (5-year horizon)
- **Tracking** model performance using MLflow

## Dataset Features

The dataset includes comprehensive property information:
- **Location**: State, City, Locality
- **Property Details**: Type (Apartment/Villa/Independent House), BHK, Size (SqFt), Price
- **Property Characteristics**: Year Built, Age, Furnished Status, Floor Number
- **Amenities**: Schools, Hospitals, Public Transport, Parking, Security, Facilities
- **Other**: Facing Direction, Owner Type, Availability Status

## Project Structure

```
real-estate-recommendation/
├── Real_estate_recommendation.ipynb  # Main notebook
├── cleaned_housing_data.csv          # Preprocessed dataset
├── mlruns/                            # MLflow tracking directory
└── README.md                          # This file
```

## Models Implemented

### Classification Models (Good Investment Prediction)
1. **Logistic Regression**
   - Simple baseline model
   - Max iterations: 1000
   
2. **Random Forest Classifier**
   - Ensemble method
   - 200 estimators, max depth: 8
   
3. **XGBoost Classifier**
   - Gradient boosting
   - 300 estimators, max depth: 5, learning rate: 0.1

### Regression Models (Future Price Prediction)
1. **Linear Regression**
   - Simple baseline model
   
2. **Random Forest Regressor**
   - 200 estimators, max depth: 8
   
3. **XGBoost Regressor**
   - 300 estimators, max depth: 5, learning rate: 0.1

## Performance Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- ROC-AUC Score

**Best Model Performance (Logistic Regression)**:
- Accuracy: 98.10%
- Precision: 97.97%
- Recall: 97.78%
- ROC-AUC: 99.85%

### Regression Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/real-estate-recommendation.git
cd real-estate-recommendation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib scikit-learn xgboost mlflow
```

## Usage

1. **Run the Jupyter Notebook**:
```bash
jupyter notebook Real_estate_recommendation.ipynb
```

2. **View MLflow Experiments**:
```bash
mlflow ui
```
Then navigate to `http://localhost:5000` to view experiment tracking.

## Key Findings

- The Logistic Regression model achieved exceptional performance (98%+ accuracy) for investment classification
- XGBoost models provide robust predictions for both classification and regression tasks
- Property location, size, and amenities are key factors in determining investment potential

## Technologies Used

- **Python 3.10**
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models and metrics
- **XGBoost**: Gradient boosting framework
- **MLflow**: Experiment tracking and model management
- **Matplotlib**: Data visualization

## Future Enhancements

- [ ] Feature engineering for better predictions
- [ ] Hyperparameter tuning using Grid/Random Search
- [ ] Model deployment using Flask/FastAPI
- [ ] Web interface for property recommendations
- [ ] Time series analysis for price trends
- [ ] Integration with real estate APIs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Always consult with real estate professionals before making investment decisions.
