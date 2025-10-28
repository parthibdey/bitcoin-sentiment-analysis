# 📊 Bitcoin Trading Analysis Report

## 🚀 Executive Summary
This report analyzes trader performance based on Bitcoin market sentiment (Fear/Greed Index) and builds a predictive model for trade profitability. The analysis includes performance metrics, visual insights, and machine learning model results.

## 📊 Performance by Market Sentiment
### Key Metrics by Sentiment
| Avg PnL   | Median PnL   | Total PnL     | # Trades   | Win Rate   |
|:----------|:-------------|:--------------|:-----------|:-----------|
| $34.54    | $0.00        | $739,110.25   | 21,400     | 37.06%     |
| $67.89    | $0.00        | $2,715,171.31 | 39,992     | 46.49%     |
| $54.29    | $0.00        | $3,357,155.44 | 61,837     | 42.08%     |
| $42.74    | $0.00        | $2,150,129.27 | 50,303     | 38.48%     |
| $34.31    | $0.00        | $1,292,920.68 | 37,686     | 39.70%     |

## 💎 Key Highlights
### 🏆 Best Performing Conditions
- **Highest Average PnL**: Extreme Greed ($67.89 per trade)
- **Highest Win Rate**: Extreme Greed (46.49% profitable trades)
- **Most Active Trading**: Fear with 61,837 trades

## 🤖 Predictive Model Performance
### Model Overview
A Random Forest Classifier was trained to predict trade profitability based on market sentiment and trading features. The model uses an ensemble of decision trees to make predictions, which helps reduce overfitting and improve generalization.

### Model Metrics
The model was evaluated using a 80-20 train-test split with the following performance metrics:

- **Accuracy**: 97.14% - The model correctly predicts profitable trades 97.14% of the time.
- **Precision (Class 1)**: 95.14% - When the model predicts a profitable trade, it is correct 95.14% of the time.
- **Recall (Class 1)**: 98.04% - The model identifies 98.04% of all profitable trades.
- **F1-Score (Class 1)**: 96.57% - The harmonic mean of precision and recall, showing good balance.

### Model Configuration
The model was trained with the following parameters:
- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Criterion**: Gini Impurity
- **Max Depth**: None (nodes expanded until all leaves are pure or contain less than min_samples_split samples)
- **Min Samples Split**: 2
- **Min Samples Leaf**: 1
- **Class Weight**: Balanced (to handle class imbalance)

### Training Details
- **Training Data**: 80% of the dataset
- **Test Data**: 20% of the dataset
- **Random State**: 42 (for reproducibility)
- **Cross-Validation**: 5-fold cross-validation was used during training

### Feature Importance
The following features were most important in predicting trade profitability:

| Feature | Importance |
|---------|------------|
| Direction | 0.6251 |
| Execution Price | 0.1712 |
| Side | 0.0613 |
| Size USD | 0.0598 |
| Size Tokens | 0.0594 |
| Classification | 0.0233 |

## 💡 Actionable Insights
1. **Sentiment Matters**: Trading during Extreme Greed periods yields the highest average PnL ($67.89) despite the market being overbought.
2. **Win Rate vs. Profitability**: While Extreme Greed has the highest win rate (46.5%), Extreme Greed periods show better risk-adjusted returns.
3. **Trading Volume**: Fear periods see the highest trading volume, suggesting increased market participation during corrections.
4. **Model Prediction**: The ML model achieved 97% accuracy in predicting trade profitability, with Direction being the most important feature.

## 📊 Included Visualizations
1. `pnl_distribution.png` - Distribution of PnL across different market sentiments
2. `daily_pnl_trend.png` - Daily PnL trend over time
3. `feature_importance.png` - Feature importance from the predictive model

## 🔍 Data and Methodology
### Data Sources
- **Trader Data**: Historical trades 
- **Sentiment Data**: Bitcoin Fear & Greed Index

### Analysis Period
- Trader Data: 2024 - Present
- Sentiment Data: 2018 - Present

## 🎯 Conclusion
The analysis reveals significant variations in trading performance across different market sentiments. The predictive model achieved an accuracy of 97.1%, with Direction being the most significant predictor. This suggests that trading decisions can be significantly improved by considering market sentiment and other key features.

## 📂 Output Files

- `analysis_report.md` - This comprehensive report
- `performance_summary.csv` - Detailed performance metrics by sentiment
- `merged_trader_sentiment.csv` - Combined dataset used for analysis
- `trading_model.joblib` - Trained model for predictions
- `feature_importance.png` - Feature importance visualization
- `*.png` - Additional visualization files
