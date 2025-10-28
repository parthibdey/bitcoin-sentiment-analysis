# Bitcoin Trading Analysis with Market Sentiment

## 📊 Project Overview
This project analyzes the relationship between Bitcoin trading performance and market sentiment, as measured by the Fear & Greed Index. It includes data processing, visualization, and a predictive model to forecast trade profitability based on market conditions.

## ✨ Features
- **Data Analysis**: Comprehensive analysis of trading performance across different market sentiments
- **Visualizations**: Interactive charts showing PnL distribution, daily trends, and feature importance
- **Predictive Modeling**: Random Forest model with 97%+ accuracy in predicting trade outcomes
- **Detailed Reporting**: Comprehensive markdown report with key findings and insights

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bitcoin_sentiment_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

1. Place your data files in the `data/` directory:
   - `historical_trader_data.csv`
   - `fear_greed_index.csv`

2. Run the analysis:
   ```bash
   python complete_analysis.py
   ```

3. View the generated report and visualizations in the `output/` directory.

## 📁 Project Structure
```
bitcoin_sentiment_analysis/
├── data/                    # Input data files
│   ├── historical_trader_data.csv
│   └── fear_greed_index.csv
├── output/                  # Generated outputs
│   ├── analysis_report.md   # Comprehensive analysis report
│   ├── trading_model.joblib # Trained model
│   ├── *.png                # Generated visualizations
│   └── *.csv                # Processed data exports
├── complete_analysis.py     # Main analysis script
└── requirements.txt         # Python dependencies
```

## 📈 Results

### Key Findings
- **Best Performing Market Condition**: Extreme Greed (Average PnL: $67.89)
- **Highest Win Rate**: 46.49% (Extreme Greed)
- **Most Active Trading**: Fear sentiment (61,837 trades)

### Model Performance
- **Accuracy**: 97.14%
- **Precision**: 95.14%
- **Recall**: 98.04%
- **F1-Score**: 96.57%

### Top Predictive Features
1. Direction (59.76%)
2. Execution Price (18.62%)
3. Size USD (6.53%)
4. Size Tokens (6.43%)
5. Side (6.06%)

## 📦 Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- tabulate
- git-lfs (for handling large files)

## 📊 Data Description

### Input Data
- `historical_trader_data.csv`: Contains trading data with columns like:
  - `Timestamp IST`: Trade timestamp
  - `Closed PnL`: Profit/Loss for each trade
  - `Side`: Trade direction (Buy/Sell)
  - (and other trading metrics)

- `fear_greed_index.csv`: Contains sentiment data with:
  - `date`: The date of the sentiment reading
  - `classification`: Market sentiment (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)

### Generated Data
- `merged_trader_sentiment.csv`: Combined dataset with:
  - All original trading data
  - `Classification`: Added from sentiment data
  - `is_profitable`: Boolean flag (True if `Closed PnL > 0`)
  - Used to calculate win rates in the analysis

## ⚠️ Large Files Notice
This repository uses Git LFS (Large File Storage) to handle large files:
- `output/trading_model.joblib` (108.21 MB)

To clone this repository with all files, first install Git LFS:
```bash
git lfs install
git clone https://github.com/parthibdey/bitcoin-sentiment-analysis.git
```
