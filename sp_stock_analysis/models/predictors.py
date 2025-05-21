import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from typing import Tuple, List, Dict, Any

class StockPredictor:
    def __init__(self):
        # Initialize with None - models will be created adaptively based on data size
        self.models = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable for ML models"""
        # Define target: 1 if price goes up in next n days, 0 otherwise
        future_returns = df['Close'].pct_change(periods=5).shift(-5)
        y = (future_returns > 0).astype(int)
        
        # Select features for ML, excluding target and non-predictive columns
        feature_columns = [
            'RSI', 'MACD', 'MACD_Hist', 'BB_Width', 'Daily_Volatility',
            'Volume_Ratio', 'ROC5', 'ROC10', 'ROC20', 'Stoch_K', 'Stoch_D',
            'ADX', 'Daily_Return'
        ]
        
        # Add moving average crossover features
        for fast, slow in [(5,20), (20,50), (50,200)]:
            df[f'MA_Cross_{fast}_{slow}'] = (
                df[f'SMA{fast}'] > df[f'SMA{slow}']
            ).astype(int)
            feature_columns.append(f'MA_Cross_{fast}_{slow}')
        
        # Store feature columns for prediction
        self.feature_columns = feature_columns
        
        # Create feature matrix
        X = df[feature_columns].copy()
        
        # Handle missing values more robustly
        X = X.fillna(method='ffill')
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col].fillna(median_val, inplace=True)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        X = X.fillna(0)  # Final fallback
        
        return X, y
        
    def create_models(self, data_size: int) -> Dict[str, Any]:
        """Create models adapted to the size of available data"""
        if data_size < 50:
            # Very small dataset - use simple models
            return {
                'lr': LogisticRegression(
                    random_state=42,
                    max_iter=100,
                    solver='liblinear',
                    C=0.1  # More regularization
                )
            }
        elif data_size < 100:
            # Small dataset - use regularized models
            return {
                'rf': RandomForestClassifier(
                    n_estimators=20,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                ),
                'lr': LogisticRegression(
                    random_state=42,
                    max_iter=200,
                    solver='liblinear',
                    C=0.1
                )
            }
        else:
            # Larger dataset - use full ensemble
            models = {
                'rf': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'lr': LogisticRegression(
                    random_state=42,
                    max_iter=500,
                    solver='liblinear',
                    C=1.0
                )
            }
            
            # Add XGBoost for larger datasets
            try:
                models['xgb'] = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except Exception:
                pass
            
            return models

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train multiple models and return their performance metrics"""
        # Remove any rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 20:
            raise ValueError("Insufficient data for ML training (need at least 20 samples)")

        # Create models based on data size
        self.models = self.create_models(len(X))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize TimeSeriesSplit with adaptive number of splits
        n_splits = min(5, len(X) // 10)  # Ensure enough samples per split
        tscv = TimeSeriesSplit(n_splits=max(2, n_splits))
        
        # Track performance of each model
        model_scores = {}
        
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            model_scores[name] = np.mean(scores)
        
        # Select best model
        self.best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[self.best_model_name]
        
        # Final fit with all data
        self.best_model.fit(X_scaled, y)
        
        return model_scores
    
    def predict_probability(self, X: pd.DataFrame) -> float:
        """Predict probability of price increase"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure we have all required features
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Prepare features
        X = X[self.feature_columns].copy()
        X = X.fillna(method='ffill').fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get probability of price increase
        probabilities = self.best_model.predict_proba(X_scaled)
        return probabilities[-1][1]  # Return probability of positive class for latest data point

    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get predictions from all models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Dict[str, float]: Dictionary of model name to prediction probability
        """
        if self.models is None:
            raise ValueError("Models have not been trained yet")
        
        # Ensure we have all required features
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Prepare features
        X = X[self.feature_columns].copy()
        X = X.fillna(method='ffill').fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[-1][1]  # Get probability of positive class
            predictions[name] = prob
        
        return predictions

def get_trading_signals(df: pd.DataFrame, probability: float, threshold: float = 0.6) -> str:
    """Generate trading signals based on ML probability and technical indicators"""
    if probability >= threshold:
        return "BUY"
    elif probability <= (1 - threshold):
        return "SELL"
    else:
        return "HOLD"
