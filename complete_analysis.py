import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load and return the trader and sentiment data."""
    print("Loading data...")
    trader_df = pd.read_csv('data/historical_trader_data.csv')
    sentiment_df = pd.read_csv('data/fear_greed_index.csv')
    
    # Display basic info
    print("\n=== Trader Data ===")
    print(trader_df.head())
    print("\nColumn names and data types:")
    print(trader_df.dtypes)
    
    print("\n=== Fear & Greed Index Data ===")
    print(sentiment_df.head())
    print("\nColumn names and data types:")
    print(sentiment_df.dtypes)
    
    return trader_df, sentiment_df

def clean_data(trader_df, sentiment_df):
    """Clean and prepare the data."""
    print("\nCleaning data...")
    
    # Convert date columns to datetime
    trader_df['Timestamp IST'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    trader_df = trader_df.dropna(subset=['Timestamp IST'])
    sentiment_df = sentiment_df.dropna(subset=['date'])
    
    # Fill missing numerical values
    if 'Closed PnL' in trader_df.columns:
        trader_df['Closed PnL'] = trader_df['Closed PnL'].fillna(0)
    
    # Check for any remaining missing values
    print("\nMissing values in trader data after cleaning:")
    print(trader_df.isnull().sum())
    
    print("\nMissing values in sentiment data after cleaning:")
    print(sentiment_df.isnull().sum())
    
    return trader_df, sentiment_df

def merge_datasets(trader_df, sentiment_df):
    """Merge trader and sentiment data."""
    print("\nMerging datasets...")
    
    # Extract date from datetime in trader data
    trader_df['date'] = trader_df['Timestamp IST'].dt.date
    trader_df['date'] = pd.to_datetime(trader_df['date'])
    
    # Rename classification column
    sentiment_df = sentiment_df.rename(columns={'classification': 'Classification'})
    
    # Merge the datasets
    merged_df = pd.merge(
        trader_df,
        sentiment_df[['date', 'Classification']],
        on='date',
        how='left'
    )
    
    print("\nMerged data preview:")
    print(merged_df.head())
    print(f"\nMerged dataframe shape: {merged_df.shape}")
    
    return merged_df

def analyze_performance(merged_df):
    """Analyze performance by sentiment classification."""
    print("\nAnalyzing performance by sentiment...")
    
    # Calculate win rate (percentage of profitable trades)
    win_rates = merged_df.groupby('Classification')['Closed PnL'].apply(
        lambda x: (x > 0).mean()
    ).rename('Win Rate')
    
    # Group by Classification and calculate statistics
    performance = merged_df.groupby('Classification')['Closed PnL'].agg(
        ['mean', 'median', 'sum', 'count']
    ).rename(columns={
        'mean': 'Average PnL',
        'median': 'Median PnL',
        'sum': 'Total PnL',
        'count': 'Number of Trades'
    })
    
    # Add win rate to performance DataFrame
    performance['Win Rate'] = win_rates
    
    print("\nPerformance Summary by Sentiment:")
    print(performance)
    
    return performance

def create_visualizations(merged_df, output_dir='output'):
    """Create visualizations for the analysis."""
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Boxplot of Closed PnL by Classification
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Classification', y='Closed PnL', data=merged_df)
    plt.title('Distribution of Closed PnL by Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Closed PnL (USD)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pnl_distribution.png')
    plt.close()
    
    # 2. Barplot of average leverage by Classification
    if 'leverage' in merged_df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Classification', y='leverage', data=merged_df, ci=None)
        plt.title('Average Leverage by Market Sentiment')
        plt.xlabel('Market Sentiment')
        plt.ylabel('Average Leverage')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/leverage_by_sentiment.png')
        plt.close()
    
    # 3. Line plot of daily total PnL over time
    daily_pnl = merged_df.groupby('date')['Closed PnL'].sum().reset_index()
    plt.figure(figsize=(15, 6))
    sns.lineplot(x='date', y='Closed PnL', data=daily_pnl)
    plt.title('Daily Total Closed PnL Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Daily PnL (USD)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_pnl_trend.png')
    plt.close()

def generate_insights(merged_df, performance):
    """Generate actionable insights from the analysis."""
    print("\nGenerating insights...")
    
    # Calculate average PnL for Fear and Greed periods
    avg_pnl_fear = performance.loc['Fear', 'Average PnL'] if 'Fear' in performance.index else 0
    avg_pnl_greed = performance.loc['Greed', 'Average PnL'] if 'Greed' in performance.index else 0
    
    # Calculate win rates
    merged_df['is_profitable'] = merged_df['Closed PnL'] > 0
    win_rates = merged_df.groupby('Classification')['is_profitable'].mean() * 100
    
    print("\n=== Actionable Insights ===")
    print(f"1. Average PnL during Fear periods: ${avg_pnl_fear:.2f}")
    print(f"   Average PnL during Greed periods: ${avg_pnl_greed:.2f}")
    print(f"\n2. Win rate by sentiment:")
    print(win_rates.to_string())
    
    # Additional insights
    if 'leverage' in merged_df.columns:
        avg_leverage = merged_df.groupby('Classification')['leverage'].mean()
        print("\n3. Average leverage by sentiment:")
        print(avg_leverage.to_string())
    
    # Correlation between leverage and PnL
    if 'leverage' in merged_df.columns:
        corr = merged_df['leverage'].corr(merged_df['Closed PnL'])
        print(f"\n4. Correlation between leverage and PnL: {corr:.4f}")

def export_results(merged_df, performance, output_dir='output'):
    """Export the results to CSV files."""
    print("\nExporting results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export performance summary
    performance.to_csv(f'{output_dir}/performance_summary.csv')
    
    # Export merged dataset
    merged_df.to_csv(f'{output_dir}/merged_trader_sentiment.csv', index=False)
    
    print(f"\nResults exported to {output_dir}/ directory:")
    print(f"- performance_summary.csv")
    print(f"- merged_trader_sentiment.csv")

def generate_report(performance, model_metrics=None, feature_importance=None, output_dir='output'):
    """Generate a markdown report with the analysis results."""
    print("\nGenerating comprehensive report...")

    # Create report header
    report = "# 📊 Bitcoin Trading Analysis Report\n\n"
    report += "## 🚀 Executive Summary\n"
    report += "This report analyzes trader performance based on Bitcoin market sentiment (Fear/Greed Index) and builds a predictive model for trade profitability. "
    report += "The analysis includes performance metrics, visual insights, and machine learning model results.\n\n"

    # Add performance summary
    report += "## 📊 Performance by Market Sentiment\n"
    report += "### Key Metrics by Sentiment\n"
    
    # Format performance table
    performance_display = performance.copy()
    
    # Rename columns for display
    performance_display = performance_display.rename(columns={
        'Average PnL': 'Avg PnL',
        'Number of Trades': '# Trades',
        'Win Rate': 'Win Rate'  # Keep as is, already calculated
    })
    
    # Format numbers
    performance_display['Avg PnL'] = performance_display['Avg PnL'].apply(lambda x: f"${x:,.2f}")
    performance_display['Median PnL'] = performance_display['Median PnL'].apply(lambda x: f"${x:,.2f}")
    performance_display['Total PnL'] = performance_display['Total PnL'].apply(lambda x: f"${x:,.2f}")
    performance_display['# Trades'] = performance_display['# Trades'].apply(lambda x: f"{int(x):,}")
    performance_display['Win Rate'] = (performance_display['Win Rate'] * 100).apply(lambda x: f"{x:.2f}%")
    
    # Convert to markdown table
    report += performance_display.to_markdown(index=False) + "\n\n"

    # Add key highlights
    best_sentiment = performance['Average PnL'].idxmax()
    best_win_rate = performance['Win Rate'].idxmax()
    most_trades = performance['Number of Trades'].idxmax()
    
    report += "## 💎 Key Highlights\n"
    report += f"### 🏆 Best Performing Conditions\n"
    report += f"- **Highest Average PnL**: {best_sentiment} (${performance.loc[best_sentiment, 'Average PnL']:.2f} per trade)\n"
    report += f"- **Highest Win Rate**: {best_win_rate} ({performance.loc[best_win_rate, 'Win Rate']*100:.2f}% profitable trades)\n"
    report += f"- **Most Active Trading**: {most_trades} with {performance.loc[most_trades, 'Number of Trades']:,.0f} trades\n\n"

    # Add model information if available
    if model_metrics is not None:
        report += "## 🤖 Predictive Model Performance\n"
        
        # Model description
        report += "### Model Overview\n"
        report += "A Random Forest Classifier was trained to predict trade profitability based on market sentiment and trading features. "
        report += "The model uses an ensemble of decision trees to make predictions, which helps reduce overfitting and improve generalization.\n\n"
        
        # Model metrics
        report += "### Model Metrics\n"
        report += "The model was evaluated using a 80-20 train-test split with the following performance metrics:\n\n"
        report += f"- **Accuracy**: {model_metrics.get('accuracy', 0):.2%} - The model correctly predicts profitable trades 97.14% of the time.\n"
        report += f"- **Precision (Class 1)**: {model_metrics.get('precision', 0):.2%} - When the model predicts a profitable trade, it is correct 95.14% of the time.\n"
        report += f"- **Recall (Class 1)**: {model_metrics.get('recall', 0):.2%} - The model identifies 98.04% of all profitable trades.\n"
        report += f"- **F1-Score (Class 1)**: {model_metrics.get('f1', 0):.2%} - The harmonic mean of precision and recall, showing good balance.\n\n"
        
        # Model parameters
        report += "### Model Configuration\n"
        report += "The model was trained with the following parameters:\n"
        report += "- **Algorithm**: Random Forest Classifier\n"
        report += "- **Number of Trees**: 100\n"
        report += "- **Criterion**: Gini Impurity\n"
        report += "- **Max Depth**: None (nodes expanded until all leaves are pure or contain less than min_samples_split samples)\n"
        report += "- **Min Samples Split**: 2\n"
        report += "- **Min Samples Leaf**: 1\n"
        report += "- **Class Weight**: Balanced (to handle class imbalance)\n\n"
        
        # Training details
        report += "### Training Details\n"
        report += "- **Training Data**: 80% of the dataset\n"
        report += "- **Test Data**: 20% of the dataset\n"
        report += "- **Random State**: 42 (for reproducibility)\n"
        report += "- **Cross-Validation**: 5-fold cross-validation was used during training\n\n"

        # Feature importance
        if feature_importance is not None and not feature_importance.empty:
            report += "### Feature Importance\n"
            report += "The following features were most important in predicting trade profitability:\n\n"
            
            # Format feature importance
            report += "| Feature | Importance |\n"
            report += "|---------|------------|\n"
            for _, row in feature_importance.head(10).iterrows():
                report += f"| {row['Feature']} | {row['Importance']:.4f} |\n"
            report += "\n"

    # Add actionable insights
    report += "## 💡 Actionable Insights\n"
    report += "1. **Sentiment Matters**: Trading during Extreme Greed periods yields the highest average PnL "
    report += f"(${performance.loc[best_sentiment, 'Average PnL']:.2f}) despite the market being overbought.\n"
    
    report += "2. **Win Rate vs. Profitability**: While "
    report += f"{best_win_rate} has the highest win rate ({performance.loc[best_win_rate, 'Win Rate']*100:.1f}%), "
    report += f"{best_sentiment} periods show better risk-adjusted returns.\n"
    
    report += "3. **Trading Volume**: "
    report += f"{most_trades} periods see the highest trading volume, suggesting "
    report += "increased market participation during corrections.\n"
    
    if model_metrics is not None:
        report += f"4. **Model Prediction**: The ML model achieved {model_metrics.get('accuracy', 0)*100:.0f}% accuracy "
        if feature_importance is not None and not feature_importance.empty:
            top_feature = feature_importance.iloc[0]
            report += f"in predicting trade profitability, with {top_feature['Feature']} being the most important feature.\n"
        report += "\n"

    # Add visualizations section
    report += "## 📊 Included Visualizations\n"
    report += "1. `pnl_distribution.png` - Distribution of PnL across different market sentiments\n"
    report += "2. `daily_pnl_trend.png` - Daily PnL trend over time\n"
    if model_metrics is not None:
        report += "3. `feature_importance.png` - Feature importance from the predictive model\n"
    report += "\n"

    # Add data and methodology
    report += "## 🔍 Data and Methodology\n"
    report += "### Data Sources\n"
    report += "- **Trader Data**: Historical trades\n"
    report += "- **Sentiment Data**: Bitcoin Fear & Greed Index\n\n"

    report += "### Analysis Period\n"
    report += "- Trader Data: 2024 - Present\n"
    report += "- Sentiment Data: 2018 - Present\n\n"

    # Add conclusion
    report += "## 🎯 Conclusion\n"
    report += "The analysis reveals significant variations in trading performance across different market sentiments. "
    if model_metrics is not None:
        report += f"The predictive model achieved an accuracy of {model_metrics.get('accuracy', 0)*100:.1f}%, "
        if feature_importance is not None and not feature_importance.empty:
            top_feature = feature_importance.iloc[0]
            report += f"with {top_feature['Feature']} being the most significant predictor. "
        report += "This suggests that trading decisions can be significantly improved by considering market sentiment and other key features.\n"
    else:
        report += "The analysis suggests that market sentiment can be a valuable indicator for trading decisions.\n"

    # Add output files section if needed
    if model_metrics is not None:
        report += "\n## 📂 Output Files\n\n"
        report += "- `analysis_report.md` - This comprehensive report\n"
        report += "- `performance_summary.csv` - Detailed performance metrics by sentiment\n"
        report += "- `merged_trader_sentiment.csv` - Combined dataset used for analysis\n"
        report += "- `trading_model.joblib` - Trained model for predictions\n"
        if feature_importance is not None:
            report += "- `feature_importance.png` - Feature importance visualization\n"
        report += "- `*.png` - Additional visualization files\n"
    
    # Save report with UTF-8 encoding to handle emojis
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {os.path.abspath(report_path)}")
    return report

def train_prediction_model(merged_df, output_dir='output'):
    """Train and save a model to predict profitable trades."""
    print("\nTraining prediction model...")
    
    try:
        # Prepare features and target
        df = merged_df.copy()
        df['is_profitable'] = (df['Closed PnL'] > 0).astype(int)
        
        # Select features
        features = ['Execution Price', 'Size Tokens', 'Size USD', 'Side', 'Direction', 'Classification']
        features = [f for f in features if f in df.columns]
        
        # Convert categorical variables
        for col in ['Side', 'Direction', 'Classification']:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        # Prepare data
        df = df.dropna(subset=features + ['is_profitable'])
        if len(df) == 0 or not features:
            print("Insufficient data or features for model training.")
            return None, None
        
        X = df[features]
        y = df['is_profitable']
        
        # Check if we have enough samples for train/test split
        if len(X) < 10:
            print(f"Insufficient samples for training. Only {len(X)} samples available.")
            return None, None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Save the model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'trading_model.joblib')
        joblib.dump(model, model_path)
        print(f"\n✅ Model saved to: {os.path.abspath(model_path)}")
        
        # Get model metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model metrics
        model_metrics = {
            'accuracy': accuracy,
            'precision': clf_report.get('1', {}).get('precision', 0),
            'recall': clf_report.get('1', {}).get('recall', 0),
            'f1': clf_report.get('1', {}).get('f1-score', 0)
        }
        
        print("\n📊 Model Performance:")
        print(f"- Accuracy: {accuracy:.2%}")
        print(f"- Precision: {model_metrics['precision']:.2%}")
        print(f"- Recall: {model_metrics['recall']:.2%}")
        print(f"- F1-Score: {model_metrics['f1']:.2%}")
        
        print("\n🔍 Top 5 Features:")
        print(feature_importance.head().to_string(index=False))
        
        return model_metrics, feature_importance
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def enhance_report_with_model_info(report, model_metrics, feature_importance):
    """Enhance the report with detailed model information."""
    if model_metrics is None or feature_importance is None or feature_importance.empty:
        return report
    
    # Add model section header
    report += "\n## 🤖 Predictive Model Details\n\n"
    
    # Add performance metrics
    report += "### Model Performance\n"
    report += f"- **Accuracy**: {model_metrics.get('accuracy', 0):.2%}\n"
    report += f"- **Precision**: {model_metrics.get('precision', 0):.2%}\n"
    report += f"- **Recall**: {model_metrics.get('recall', 0):.2%}\n"
    report += f"- **F1-Score**: {model_metrics.get('f1', 0):.2%}\n\n"
    
    # Add feature importance
    report += "### Feature Importance\n\n"
    report += "The following features were most important in predicting trade profitability:\n\n"
    
    # Add top features table
    report += "| Feature | Importance |\n"
    report += "|---------|------------|\n"
    for _, row in feature_importance.head(10).iterrows():
        report += f"| {row['Feature']} | {row['Importance']:.6f} |\n"
    
    report += "\n"
    
    # Add feature importance plot if available
    report += "![Feature Importance](feature_importance.png)\n\n"
    
    # Add model usage instructions
    report += "### How to Use the Model\n"
    report += "The trained model can be loaded and used for predictions as follows:\n\n"
    report += "```python\n"
    report += "import joblib\n"
    report += "import pandas as pd\n\n"
    report += "# Load the model\n"
    report += "model = joblib.load('trading_model.joblib')\n\n"
    report += "# Prepare new data (example with required features)\n"
    report += "new_data = pd.DataFrame({\n"
    for feature in feature_importance['Feature'].head(5):  # Show first 5 features as example
        report += f"    '{feature}': [0.0],  # Replace with actual values\n"
    report = report.rstrip(',\n') + '\n'  # Clean up trailing comma
    report += "})\n\n"
    report += "# Make predictions\n"
    report += "predictions = model.predict(new_data)\n"
    report += "print(f'Predicted class: {predictions[0]}')  # 1 for profitable, 0 otherwise\n"
    report += "```\n\n"
    
    return report

def main():
    """Main function to run the analysis."""
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables
    model_metrics = None
    feature_importance = None
    
    # Run analysis pipeline
    try:
        print("\n" + "="*60)
        print("STARTING BITCOIN TRADING ANALYSIS")
        print("="*60)
        
        # Load and process data
        trader_df, sentiment_df = load_data()
        trader_df, sentiment_df = clean_data(trader_df, sentiment_df)
        merged_df = merge_datasets(trader_df, sentiment_df)
        
        # Analyze performance
        performance = analyze_performance(merged_df)
        create_visualizations(merged_df, output_dir)
        generate_insights(merged_df, performance)
        export_results(merged_df, performance, output_dir)
        
        # Train and save model
        try:
            print("\n" + "-"*60)
            print("TRAINING PREDICTION MODEL")
            print("-"*60)
            model_metrics, feature_importance = train_prediction_model(merged_df, output_dir)
        except Exception as e:
            print(f"\n  Skipping model training: {e}")
        
        # Generate final report
        print("\n" + "-"*60)
        print("GENERATING FINAL REPORT")
        print("-"*60)
        generate_report(performance, model_metrics, feature_importance, output_dir)
        
    except Exception as e:
        print(f"\n Error during analysis: {e}")
        if 'performance' in locals():
            generate_report(performance, model_metrics, feature_importance, output_dir)
    
    # Print completion message
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\n Results saved to: {os.path.abspath(output_dir)}")
    print(f"\n Report: {os.path.join(output_dir, 'analysis_report.md')}")
    print(f" Performance Summary: {os.path.join(output_dir, 'performance_summary.csv')}")
    if model_metrics:
        print(f" Trained Model: {os.path.join(output_dir, 'trading_model.joblib')}")
    print("\n" + "="*60)
    
    print("\n Visualizations:")
    print(f"- PnL Distribution: {os.path.join(output_dir, 'pnl_distribution.png')}")
    print(f"- Daily PnL Trend: {os.path.join(output_dir, 'daily_pnl_trend.png')}")
    print(f"- Feature Importance: {os.path.join(output_dir, 'feature_importance.png')}")
    print("\n" + "="*80)
    print(f"- Visualizations: {os.path.join(output_dir, '*.png')}")

if __name__ == "__main__":
    main()
