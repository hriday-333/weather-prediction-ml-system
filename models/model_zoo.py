"""
Comprehensive Model Zoo for Weather Prediction
Includes multiple regression models, time-series models, and evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet
import warnings
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import yaml
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherModelZoo:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the model zoo with configuration"""
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            self.models = {}
            self.results = {}
            self.scalers = {}
            self.trained_models = {}
            
            # Create output directories
            self.model_dir = "models/trained"
            self.results_dir = "models/results"
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            
            logger.info("Model zoo initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model zoo: {e}")
            raise
    
    def _initialize_models(self) -> Dict:
        """Initialize all regression models"""
        try:
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'Lasso': Lasso(alpha=1.0, random_state=42),
                'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'DecisionTree': DecisionTreeRegressor(random_state=42),
                'KNeighbors': KNeighborsRegressor(n_neighbors=5),
                'SVR': SVR(kernel='rbf', C=1.0),
                'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
                'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            }
            
            logger.info(f"Initialized {len(models)} regression models")
            return models
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {
                'MAE': mean_absolute_error(y_true, y_pred),
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred),
                'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
            }
            
            # Additional custom metrics
            metrics['Mean_Error'] = np.mean(y_pred - y_true)
            metrics['Std_Error'] = np.std(y_pred - y_true)
            metrics['Max_Error'] = np.max(np.abs(y_pred - y_true))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2, city_filter: Optional[str] = None) -> Tuple:
        """Prepare data for model training"""
        try:
            # Filter by city if specified
            if city_filter:
                df = df[df['city'] == city_filter].copy()
                logger.info(f"Filtered data for city: {city_filter}")
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Select features (exclude non-predictive columns)
            exclude_cols = ['datetime', 'city', 'state', target_column]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Handle categorical variables
            categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns
            df_processed = df.copy()
            
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            
            # Prepare features and target
            X = df_processed[feature_cols]
            y = df_processed[target_column]
            
            # Handle missing values
            X = X.fillna(X.median())
            y = y.fillna(y.median())
            
            # Time series split (maintain temporal order)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[target_column] = scaler
            
            logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Features: {len(feature_cols)}, Target: {target_column}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    target_name: str) -> Dict[str, Any]:
        """Train all models and return results"""
        try:
            models = self._initialize_models()
            results = {}
            
            logger.info(f"Training {len(models)} models for target: {target_name}")
            
            for model_name, model in models.items():
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Train model
                    start_time = datetime.now()
                    model.fit(X_train, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Store trained model
                    model_key = f"{target_name}_{model_name}"
                    self.trained_models[model_key] = model
                    
                    # Save model
                    model_path = f"{self.model_dir}/{model_key}.joblib"
                    joblib.dump(model, model_path)
                    
                    results[model_name] = {
                        'model': model,
                        'training_time': training_time,
                        'model_path': model_path,
                        'status': 'success'
                    }
                    
                    logger.info(f"{model_name} trained successfully in {training_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    results[model_name] = {
                        'model': None,
                        'training_time': 0,
                        'model_path': None,
                        'status': 'failed',
                        'error': str(e)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise
    
    def evaluate_models(self, trained_models: Dict, X_test: np.ndarray, 
                       y_test: np.ndarray, target_name: str) -> pd.DataFrame:
        """Evaluate all trained models"""
        try:
            evaluation_results = []
            
            logger.info(f"Evaluating models for target: {target_name}")
            
            for model_name, model_info in trained_models.items():
                if model_info['status'] == 'success':
                    try:
                        model = model_info['model']
                        
                        # Make predictions
                        start_time = datetime.now()
                        y_pred = model.predict(X_test)
                        prediction_time = (datetime.now() - start_time).total_seconds()
                        
                        # Calculate metrics
                        metrics = self._calculate_metrics(y_test, y_pred)
                        
                        # Add model info
                        result = {
                            'Model': model_name,
                            'Target': target_name,
                            'Training_Time': model_info['training_time'],
                            'Prediction_Time': prediction_time,
                            **metrics
                        }
                        
                        evaluation_results.append(result)
                        
                        logger.info(f"{model_name} - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {model_name}: {e}")
                        
            # Create results DataFrame
            results_df = pd.DataFrame(evaluation_results)
            
            if not results_df.empty:
                # Sort by R² score (descending)
                results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)
                
                # Add ranking
                results_df['Rank'] = range(1, len(results_df) + 1)
                
                # Save results
                results_path = f"{self.results_dir}/{target_name}_model_results.csv"
                results_df.to_csv(results_path, index=False)
                logger.info(f"Results saved: {results_path}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            raise
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                             model_name: str, target_name: str) -> Dict:
        """Perform hyperparameter tuning for specific models"""
        try:
            logger.info(f"Hyperparameter tuning for {model_name}")
            
            # Define parameter grids
            param_grids = {
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'LightGBM': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
            
            if model_name not in param_grids:
                logger.warning(f"No parameter grid defined for {model_name}")
                return {}
            
            # Initialize model
            models_map = {
                'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0)
            }
            
            model = models_map[model_name]
            param_grid = param_grids[model_name]
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Results
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'best_model': grid_search.best_estimator_
            }
            
            # Save tuned model
            tuned_model_path = f"{self.model_dir}/{target_name}_{model_name}_tuned.joblib"
            joblib.dump(grid_search.best_estimator_, tuned_model_path)
            
            logger.info(f"Hyperparameter tuning completed for {model_name}")
            logger.info(f"Best score: {tuning_results['best_score']:.4f}")
            
            return tuning_results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise
    
    def create_model_comparison_plots(self, results_df: pd.DataFrame, target_name: str) -> None:
        """Create comprehensive model comparison visualizations"""
        try:
            if results_df.empty:
                logger.warning("No results to plot")
                return
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Score Comparison', 'RMSE Comparison', 
                              'Training Time', 'MAE vs MAPE'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # R² Score
            fig.add_trace(
                go.Bar(x=results_df['Model'], y=results_df['R2'], 
                      name='R² Score', marker_color='lightblue'),
                row=1, col=1
            )
            
            # RMSE
            fig.add_trace(
                go.Bar(x=results_df['Model'], y=results_df['RMSE'], 
                      name='RMSE', marker_color='lightcoral'),
                row=1, col=2
            )
            
            # Training Time
            fig.add_trace(
                go.Bar(x=results_df['Model'], y=results_df['Training_Time'], 
                      name='Training Time (s)', marker_color='lightgreen'),
                row=2, col=1
            )
            
            # MAE vs MAPE scatter
            fig.add_trace(
                go.Scatter(x=results_df['MAE'], y=results_df['MAPE'], 
                          mode='markers+text', text=results_df['Model'],
                          textposition="top center", name='MAE vs MAPE',
                          marker=dict(size=10, color='purple')),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f'Model Performance Comparison - {target_name.title()}',
                showlegend=False,
                height=800
            )
            
            # Rotate x-axis labels
            fig.update_xaxes(tickangle=45)
            
            # Save plot
            plot_path = f"{self.results_dir}/{target_name}_model_comparison.html"
            fig.write_html(plot_path)
            
            # Also create matplotlib version
            plt.figure(figsize=(16, 12))
            
            # R² Score
            plt.subplot(2, 2, 1)
            sns.barplot(data=results_df, x='Model', y='R2', palette='viridis')
            plt.title('R² Score Comparison')
            plt.xticks(rotation=45)
            plt.ylabel('R² Score')
            
            # RMSE
            plt.subplot(2, 2, 2)
            sns.barplot(data=results_df, x='Model', y='RMSE', palette='plasma')
            plt.title('RMSE Comparison')
            plt.xticks(rotation=45)
            plt.ylabel('RMSE')
            
            # Training Time
            plt.subplot(2, 2, 3)
            sns.barplot(data=results_df, x='Model', y='Training_Time', palette='coolwarm')
            plt.title('Training Time Comparison')
            plt.xticks(rotation=45)
            plt.ylabel('Training Time (seconds)')
            
            # Performance Scatter
            plt.subplot(2, 2, 4)
            plt.scatter(results_df['MAE'], results_df['MAPE'], s=100, alpha=0.7, c='purple')
            for i, model in enumerate(results_df['Model']):
                plt.annotate(model, (results_df['MAE'].iloc[i], results_df['MAPE'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            plt.xlabel('MAE')
            plt.ylabel('MAPE (%)')
            plt.title('MAE vs MAPE')
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/{target_name}_model_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison plots saved for {target_name}")
            
        except Exception as e:
            logger.error(f"Error creating comparison plots: {e}")
            raise
    
    def predict_future(self, model_name: str, target_name: str, 
                      X_recent: np.ndarray, days_ahead: int = 7) -> np.ndarray:
        """Make future predictions using trained model"""
        try:
            model_key = f"{target_name}_{model_name}"
            
            if model_key not in self.trained_models:
                # Try to load from file
                model_path = f"{self.model_dir}/{model_key}.joblib"
                if os.path.exists(model_path):
                    self.trained_models[model_key] = joblib.load(model_path)
                else:
                    raise ValueError(f"Model {model_key} not found")
            
            model = self.trained_models[model_key]
            
            # Make predictions
            predictions = []
            current_features = X_recent[-1:].copy()  # Use most recent features
            
            for _ in range(days_ahead * 24):  # Hourly predictions for specified days
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified approach)
                # In practice, you'd update time-based features and lag features
                current_features = current_features.copy()
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error making future predictions: {e}")
            raise
    
    def get_feature_importance(self, model_name: str, target_name: str, 
                              feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        try:
            model_key = f"{target_name}_{model_name}"
            
            if model_key not in self.trained_models:
                model_path = f"{self.model_dir}/{model_key}.joblib"
                if os.path.exists(model_path):
                    self.trained_models[model_key] = joblib.load(model_path)
                else:
                    raise ValueError(f"Model {model_key} not found")
            
            model = self.trained_models[model_key]
            
            # Check if model has feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                return importance_df
            else:
                logger.warning(f"Model {model_name} does not have feature importance")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise
    
    def run_complete_analysis(self, df: pd.DataFrame, target_columns: List[str], 
                             city_filter: Optional[str] = None) -> Dict:
        """Run complete model analysis for all targets"""
        try:
            logger.info("Starting complete model analysis...")
            
            all_results = {}
            
            for target in target_columns:
                if target not in df.columns:
                    logger.warning(f"Target column {target} not found in dataset")
                    continue
                
                logger.info(f"\n{'='*50}")
                logger.info(f"ANALYZING TARGET: {target.upper()}")
                logger.info(f"{'='*50}")
                
                # Prepare data
                X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(
                    df, target, city_filter=city_filter
                )
                
                # Train models
                trained_models = self.train_models(X_train, y_train, target)
                
                # Evaluate models
                results_df = self.evaluate_models(trained_models, X_test, y_test, target)
                
                # Create comparison plots
                if not results_df.empty:
                    self.create_model_comparison_plots(results_df, target)
                
                # Hyperparameter tuning for top 3 models
                if not results_df.empty:
                    top_models = results_df.head(3)['Model'].tolist()
                    tuning_results = {}
                    
                    for model_name in top_models:
                        if model_name in ['RandomForest', 'GradientBoosting', 'LightGBM', 'XGBoost']:
                            tuning_results[model_name] = self.hyperparameter_tuning(
                                X_train, y_train, model_name, target
                            )
                
                # Store results
                all_results[target] = {
                    'results_df': results_df,
                    'trained_models': trained_models,
                    'feature_columns': feature_cols,
                    'data_shapes': {
                        'train': X_train.shape,
                        'test': X_test.shape
                    }
                }
                
                # Print summary
                if not results_df.empty:
                    best_model = results_df.iloc[0]
                    logger.info(f"\nBEST MODEL FOR {target.upper()}:")
                    logger.info(f"Model: {best_model['Model']}")
                    logger.info(f"R²: {best_model['R2']:.4f}")
                    logger.info(f"RMSE: {best_model['RMSE']:.4f}")
                    logger.info(f"MAE: {best_model['MAE']:.4f}")
            
            # Generate summary report
            self._generate_summary_report(all_results)
            
            logger.info("\nComplete model analysis finished!")
            return all_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise
    
    def _generate_summary_report(self, all_results: Dict) -> None:
        """Generate summary report of all model results"""
        try:
            report_lines = [
                "="*60,
                "WEATHER PREDICTION MODEL ANALYSIS SUMMARY",
                "="*60,
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]
            
            for target, results in all_results.items():
                results_df = results['results_df']
                if not results_df.empty:
                    best_model = results_df.iloc[0]
                    report_lines.extend([
                        f"TARGET: {target.upper()}",
                        f"  Best Model: {best_model['Model']}",
                        f"  R² Score: {best_model['R2']:.4f}",
                        f"  RMSE: {best_model['RMSE']:.4f}",
                        f"  MAE: {best_model['MAE']:.4f}",
                        f"  Training Time: {best_model['Training_Time']:.2f}s",
                        f"  Models Tested: {len(results_df)}",
                        ""
                    ])
            
            report_lines.extend([
                "FILES GENERATED:",
                f"  Model Files: {self.model_dir}/",
                f"  Results: {self.results_dir}/",
                f"  Plots: {self.results_dir}/",
                "="*60
            ])
            
            report_text = "\n".join(report_lines)
            
            # Save report
            report_path = f"{self.results_dir}/model_analysis_summary.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)
            
            print(report_text)
            logger.info(f"Summary report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise

def main():
    """Main function to run model analysis"""
    try:
        # Check if cleaned dataset exists
        data_path = "data/weather_dataset_cleaned.csv"
        if not os.path.exists(data_path):
            logger.error(f"Cleaned dataset not found at {data_path}. Please run EDA first.")
            return
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} records")
        
        # Initialize model zoo
        model_zoo = WeatherModelZoo()
        
        # Define target variables
        target_columns = ['temperature', 'aqi', 'pm25']
        
        # Run complete analysis
        results = model_zoo.run_complete_analysis(df, target_columns)
        
        print("\nModel Analysis Complete!")
        print(f"Results saved in: {model_zoo.results_dir}")
        print(f"Trained models saved in: {model_zoo.model_dir}")
        
    except Exception as e:
        logger.error(f"Model analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
