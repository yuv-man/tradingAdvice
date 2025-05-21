"""
Visualization module for stock analysis results.
Provides functions for creating various charts and visual analysis tools.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import pandas as pd
from datetime import datetime
import os
from typing import Tuple, Optional, Dict

def plot_recommendation_distribution(results_df: pd.DataFrame) -> None:
    """Create a bar plot showing distribution of recommendations"""
    plt.figure(figsize=(10, 6))
    
    # Define recommendation order
    rec_order = [
        'STRONG_BUY', 'BUY', 'WEAK_BUY', 
        'HOLD', 
        'WEAK_SELL', 'SELL', 'STRONG_SELL'
    ]
    
    # Only include categories that exist in the data
    rec_counts = results_df['Recommendation'].value_counts()
    existing_categories = [cat for cat in rec_order if cat in rec_counts.index]
    
    if existing_categories:
        sns.countplot(x='Recommendation', data=results_df, order=existing_categories)
        plt.title('Stock Recommendations Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No recommendation data to plot")

def create_sentiment_scatter(results_df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """Create scatter plot of Sentiment vs ML Probability"""
    if 'Sentiment' not in results_df.columns or 'ML_Probability' not in results_df.columns:
        print("Cannot create scatter plot: missing required columns (Sentiment and/or ML_Probability)")
        return
    
    try:
        plt.figure(figsize=(12, 8))
        recommendation_colors = {
            'STRONG_BUY': 'darkgreen',
            'BUY': 'green',
            'WEAK_BUY': 'lightgreen',
            'HOLD': 'blue',
            'WEAK_SELL': 'salmon',
            'SELL': 'red',
            'STRONG_SELL': 'darkred'
        }

        # Plot points by recommendation
        for rec in recommendation_colors:
            subset = results_df[results_df['Recommendation'] == rec]
            if len(subset) > 0:
                plt.scatter(subset['Sentiment'], subset['ML_Probability'], 
                          c=recommendation_colors[rec], label=rec, s=100, alpha=0.7)

        # Add stock symbols as annotations
        for i, row in results_df.iterrows():
            plt.annotate(row['Symbol'], 
                        (row['Sentiment'], row['ML_Probability']),
                        xytext=(5, 5), textcoords='offset points')

        # Add reference lines
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        plt.xlabel('Sentiment Score')
        plt.ylabel('ML Probability (Higher = More Bullish)')
        plt.title('Stock Analysis: Sentiment vs ML Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if output directory is specified
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(output_dir, f"sentiment_ml_plot_{timestamp}.png"))
        
        plt.show()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")

def style_dataframe(results_df: pd.DataFrame) -> pd.DataFrame.style:
    """Apply conditional formatting to results DataFrame"""
    def color_recommendation(val):
        if val == 'STRONG_BUY':
            return 'background-color: darkgreen; color: white'
        elif val == 'BUY':
            return 'background-color: green; color: white'
        elif val == 'WEAK_BUY':
            return 'background-color: lightgreen'
        elif val == 'HOLD':
            return 'background-color: lightblue'
        elif val == 'WEAK_SELL':
            return 'background-color: salmon'
        elif val == 'SELL':
            return 'background-color: red; color: white'
        elif val == 'STRONG_SELL':
            return 'background-color: darkred; color: white'
        return ''

    try:
        styled_df = results_df.style.applymap(color_recommendation, subset=['Recommendation'])
        return styled_df
    except Exception as e:
        print(f"Error styling DataFrame: {e}")
        return results_df

def plot_results(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create comprehensive visualization of analysis results"""
    if len(results_df) == 0:
        print("No results to visualize")
        return pd.DataFrame(), pd.DataFrame()

    # Display recommendations summary
    print("\nRecommendation Summary:")
    rec_counts = results_df['Recommendation'].value_counts()
    display(rec_counts)

    # Plot recommendation distribution
    plot_recommendation_distribution(results_df)

    # Ensure we have required columns for displays
    required_cols = ['Symbol', 'Last_Price', 'Recommendation']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    
    if missing_cols:
        print(f"Missing required columns for display: {missing_cols}")
        return results_df, results_df  # Return empty DataFrames as placeholders
        
    # Optional columns to display if available
    display_cols = ['Symbol', 'Last_Price', 'Recommendation']
    for col in ['RSI', 'ADX', 'Sentiment', 'ML_Probability', 'ML_Confidence']:
        if col in results_df.columns:
            display_cols.append(col)
    
    # Show top buy recommendations
    buy_stocks = results_df[
        results_df['Recommendation'].isin(['STRONG_BUY', 'BUY'])
    ]
    if 'ML_Probability' in results_df.columns:
        buy_stocks = buy_stocks.sort_values('ML_Probability', ascending=False)
    else:
        buy_stocks = buy_stocks.sort_values('Last_Price', ascending=True)
    
    print("\nTOP BUY RECOMMENDATIONS:")
    if len(buy_stocks) > 0:
        display(buy_stocks[display_cols])
    else:
        print("No buy recommendations found")

    # Show top sell recommendations
    sell_stocks = results_df[
        results_df['Recommendation'].isin(['STRONG_SELL', 'SELL'])
    ]
    if 'ML_Probability' in results_df.columns:
        sell_stocks = sell_stocks.sort_values('ML_Probability', ascending=True)
    else:
        sell_stocks = sell_stocks.sort_values('Last_Price', ascending=True)
    
    print("\nTOP SELL RECOMMENDATIONS:")
    if len(sell_stocks) > 0:
        display(sell_stocks[display_cols])
    else:
        print("No sell recommendations found")

    # Create sentiment scatter plot if data available
    create_sentiment_scatter(results_df)
    
    return buy_stocks, sell_stocks

def save_results_html(results_df: pd.DataFrame, output_dir: str = './results') -> str:
    """Save analysis results as HTML report"""
    if len(results_df) == 0:
        print("No results to save")
        return ""
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Count recommendations
    rec_counts = {
        'STRONG_BUY': len(results_df[results_df['Recommendation'] == 'STRONG_BUY']),
        'BUY': len(results_df[results_df['Recommendation'] == 'BUY']), 
        'HOLD': len(results_df[results_df['Recommendation'] == 'HOLD']),
        'SELL': len(results_df[results_df['Recommendation'] == 'SELL']),
        'STRONG_SELL': len(results_df[results_df['Recommendation'] == 'STRONG_SELL'])
    }
    
    # Get buy and sell recommendations
    buy_recs = results_df[results_df['Recommendation'].isin(['STRONG_BUY', 'BUY'])]
    sell_recs = results_df[results_df['Recommendation'].isin(['STRONG_SELL', 'SELL'])]
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Stock Analysis Report - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #4CAF50; color: white; }}
            .buy {{ color: green; font-weight: bold; }}
            .sell {{ color: red; font-weight: bold; }}
            .hold {{ color: blue; font-weight: bold; }}
            .summary {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Stock Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Stocks Analyzed: {len(results_df)}</p>
            <ul>
                <li>Strong Buy: {rec_counts['STRONG_BUY']}</li>
                <li>Buy: {rec_counts['BUY']}</li>
                <li>Hold: {rec_counts['HOLD']}</li>
                <li>Sell: {rec_counts['SELL']}</li>
                <li>Strong Sell: {rec_counts['STRONG_SELL']}</li>
            </ul>
        </div>
        
        <h2>Top Buy Recommendations</h2>
        {buy_recs.to_html(classes='data', index=False) if len(buy_recs) > 0 
         else "<p>No buy recommendations found.</p>"}
        
        <h2>Top Sell Recommendations</h2>
        {sell_recs.to_html(classes='data', index=False) if len(sell_recs) > 0 
         else "<p>No sell recommendations found.</p>"}
    </body>
    </html>
    """
    
    # Save HTML report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_file = os.path.join(output_dir, f"analysis_report_{timestamp}.html")
    with open(html_file, 'w') as f:
        f.write(html_report)
    
    return html_file
