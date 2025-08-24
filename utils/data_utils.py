"""
Data Utility Functions for Weather Prediction Project
Includes data manipulation, preprocessing, and helper functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Comprehensive data processing utilities"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median', 
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values using various strategies"""
        try:
            df_processed = df.copy()
            
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if strategy == 'drop':
                # Drop rows with any missing values in specified columns
                df_processed = df_processed.dropna(subset=columns)
                logger.info(f"Dropped rows with missing values. New shape: {df_processed.shape}")
                
            elif strategy in ['mean', 'median', 'most_frequent']:
                # Use SimpleImputer
                imputer = SimpleImputer(strategy=strategy)
                df_processed[columns] = imputer.fit_transform(df_processed[columns])
                self.imputers[strategy] = imputer
                logger.info(f"Filled missing values using {strategy} strategy")
                
            elif strategy == 'knn':
                # Use KNN imputation
                imputer = KNNImputer(n_neighbors=5)
                df_processed[columns] = imputer.fit_transform(df_processed[columns])
                self.imputers['knn'] = imputer
                logger.info("Filled missing values using KNN imputation")
                
            elif strategy == 'forward_fill':
                # Forward fill (useful for time series)
                df_processed[columns] = df_processed[columns].fillna(method='ffill')
                logger.info("Applied forward fill for missing values")
                
            elif strategy == 'interpolate':
                # Linear interpolation
                df_processed[columns] = df_processed[columns].interpolate(method='linear')
                logger.info("Applied linear interpolation for missing values")
                
            return df_processed
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr') -> Dict[str, np.ndarray]:
        """Detect outliers using various methods"""
        try:
            outliers = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                data = df[col].dropna()
                
                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (data < lower_bound) | (data > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((data - data.mean()) / data.std())
                    outlier_mask = z_scores > 3
                    
                elif method == 'modified_zscore':
                    median = data.median()
                    mad = np.median(np.abs(data - median))
                    modified_z_scores = 0.6745 * (data - median) / mad
                    outlier_mask = np.abs(modified_z_scores) > 3.5
                    
                outliers[col] = data[outlier_mask].index.tolist()
                logger.info(f"Detected {len(outliers[col])} outliers in {col} using {method}")
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            raise
    
    def handle_outliers(self, df: pd.DataFrame, outliers: Dict[str, List], 
                       method: str = 'cap') -> pd.DataFrame:
        """Handle outliers using various methods"""
        try:
            df_processed = df.copy()
            
            for col, outlier_indices in outliers.items():
                if not outlier_indices:
                    continue
                    
                if method == 'remove':
                    # Remove outlier rows
                    df_processed = df_processed.drop(outlier_indices)
                    
                elif method == 'cap':
                    # Cap outliers to percentile values
                    lower_cap = df_processed[col].quantile(0.05)
                    upper_cap = df_processed[col].quantile(0.95)
                    df_processed[col] = np.clip(df_processed[col], lower_cap, upper_cap)
                    
                elif method == 'transform':
                    # Log transformation for positive skewed data
                    if df_processed[col].min() > 0:
                        df_processed[col] = np.log1p(df_processed[col])
                    
                logger.info(f"Handled outliers in {col} using {method} method")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error handling outliers: {e}")
            raise
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], 
                           lags: List[int], group_col: str = 'city') -> pd.DataFrame:
        """Create lag features for time series analysis"""
        try:
            df_processed = df.copy()
            df_processed = df_processed.sort_values(['datetime', group_col])
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}h"
                    df_processed[lag_col_name] = df_processed.groupby(group_col)[col].shift(lag)
                    
            logger.info(f"Created lag features for {len(columns)} columns with lags: {lags}")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            raise
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                               windows: List[int], group_col: str = 'city') -> pd.DataFrame:
        """Create rolling window features"""
        try:
            df_processed = df.copy()
            df_processed = df_processed.sort_values(['datetime', group_col])
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                for window in windows:
                    # Rolling mean
                    rolling_mean_col = f"{col}_rolling_mean_{window}h"
                    df_processed[rolling_mean_col] = (
                        df_processed.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
                    
                    # Rolling std
                    rolling_std_col = f"{col}_rolling_std_{window}h"
                    df_processed[rolling_std_col] = (
                        df_processed.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(0, drop=True)
                    )
                    
                    # Rolling min/max
                    rolling_min_col = f"{col}_rolling_min_{window}h"
                    rolling_max_col = f"{col}_rolling_max_{window}h"
                    
                    df_processed[rolling_min_col] = (
                        df_processed.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .min()
                        .reset_index(0, drop=True)
                    )
                    
                    df_processed[rolling_max_col] = (
                        df_processed.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .max()
                        .reset_index(0, drop=True)
                    )
                    
            logger.info(f"Created rolling features for {len(columns)} columns with windows: {windows}")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
            raise
    
    def create_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
        """Create comprehensive temporal features"""
        try:
            df_processed = df.copy()
            df_processed[datetime_col] = pd.to_datetime(df_processed[datetime_col])
            
            # Basic temporal features
            df_processed['year'] = df_processed[datetime_col].dt.year
            df_processed['month'] = df_processed[datetime_col].dt.month
            df_processed['day'] = df_processed[datetime_col].dt.day
            df_processed['hour'] = df_processed[datetime_col].dt.hour
            df_processed['minute'] = df_processed[datetime_col].dt.minute
            df_processed['day_of_week'] = df_processed[datetime_col].dt.dayofweek
            df_processed['day_of_year'] = df_processed[datetime_col].dt.dayofyear
            df_processed['week_of_year'] = df_processed[datetime_col].dt.isocalendar().week
            df_processed['quarter'] = df_processed[datetime_col].dt.quarter
            
            # Cyclical features (sine/cosine encoding)
            df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
            df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
            df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
            df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
            df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
            df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
            
            # Boolean features
            df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
            df_processed['is_business_hour'] = ((df_processed['hour'] >= 9) & 
                                              (df_processed['hour'] <= 17)).astype(int)
            df_processed['is_rush_hour'] = (df_processed['hour'].isin([7, 8, 9, 17, 18, 19])).astype(int)
            
            # Season mapping
            season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                         3: 'spring', 4: 'spring', 5: 'spring',
                         6: 'summer', 7: 'summer', 8: 'summer',
                         9: 'monsoon', 10: 'monsoon', 11: 'autumn'}
            df_processed['season'] = df_processed['month'].map(season_map)
            
            # Season boolean features
            df_processed['is_winter'] = (df_processed['season'] == 'winter').astype(int)
            df_processed['is_spring'] = (df_processed['season'] == 'spring').astype(int)
            df_processed['is_summer'] = (df_processed['season'] == 'summer').astype(int)
            df_processed['is_monsoon'] = (df_processed['season'] == 'monsoon').astype(int)
            df_processed['is_autumn'] = (df_processed['season'] == 'autumn').astype(int)
            
            logger.info("Created comprehensive temporal features")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error creating temporal features: {e}")
            raise
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], 
                      method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features using various methods"""
        try:
            df_processed = df.copy()
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            df_processed[columns] = scaler.fit_transform(df_processed[columns])
            self.scalers[method] = scaler
            
            logger.info(f"Scaled features using {method} scaling")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between specified pairs"""
        try:
            df_processed = df.copy()
            
            for feat1, feat2 in feature_pairs:
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplication interaction
                    interaction_col = f"{feat1}_x_{feat2}"
                    df_processed[interaction_col] = df_processed[feat1] * df_processed[feat2]
                    
                    # Ratio interaction (if feat2 is not zero)
                    if (df_processed[feat2] != 0).all():
                        ratio_col = f"{feat1}_div_{feat2}"
                        df_processed[ratio_col] = df_processed[feat1] / df_processed[feat2]
            
            logger.info(f"Created interaction features for {len(feature_pairs)} pairs")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            raise
    
    def create_binned_features(self, df: pd.DataFrame, columns: List[str], 
                              n_bins: int = 5) -> pd.DataFrame:
        """Create binned categorical features from continuous variables"""
        try:
            df_processed = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                # Create bins
                binned_col = f"{col}_binned"
                df_processed[binned_col] = pd.cut(df_processed[col], bins=n_bins, 
                                                labels=[f"{col}_bin_{i}" for i in range(n_bins)])
                
                # Create one-hot encoding
                bin_dummies = pd.get_dummies(df_processed[binned_col], prefix=f"{col}_bin")
                df_processed = pd.concat([df_processed, bin_dummies], axis=1)
                
                # Drop the intermediate binned column
                df_processed.drop(binned_col, axis=1, inplace=True)
            
            logger.info(f"Created binned features for {len(columns)} columns")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error creating binned features: {e}")
            raise
    
    def feature_selection_correlation(self, df: pd.DataFrame, target_col: str, 
                                    threshold: float = 0.1) -> List[str]:
        """Select features based on correlation with target"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            correlations = df[numeric_cols + [target_col]].corr()[target_col].abs()
            selected_features = correlations[correlations > threshold].index.tolist()
            
            if target_col in selected_features:
                selected_features.remove(target_col)
            
            logger.info(f"Selected {len(selected_features)} features based on correlation > {threshold}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in correlation-based feature selection: {e}")
            raise
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, 
                                        threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features to reduce multicollinearity"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Find pairs of highly correlated features
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            df_processed = df.drop(columns=to_drop)
            
            logger.info(f"Removed {len(to_drop)} highly correlated features")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error removing highly correlated features: {e}")
            raise

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        try:
            quality_report = {
                'shape': df.shape,
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
            }
            
            # Check for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            infinite_values = {}
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    infinite_values[col] = inf_count
            quality_report['infinite_values'] = infinite_values
            
            # Check for constant columns
            constant_columns = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            quality_report['constant_columns'] = constant_columns
            
            # Check data ranges for key columns
            key_columns = ['temperature', 'humidity', 'pressure', 'aqi', 'pm25']
            range_issues = {}
            for col in key_columns:
                if col in df.columns:
                    col_min, col_max = df[col].min(), df[col].max()
                    
                    # Define reasonable ranges
                    reasonable_ranges = {
                        'temperature': (-50, 60),
                        'humidity': (0, 100),
                        'pressure': (900, 1100),
                        'aqi': (0, 500),
                        'pm25': (0, 1000)
                    }
                    
                    if col in reasonable_ranges:
                        min_range, max_range = reasonable_ranges[col]
                        if col_min < min_range or col_max > max_range:
                            range_issues[col] = {
                                'actual_range': (col_min, col_max),
                                'expected_range': (min_range, max_range)
                            }
            
            quality_report['range_issues'] = range_issues
            
            logger.info("Data quality validation completed")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in data quality validation: {e}")
            raise
    
    @staticmethod
    def validate_time_series(df: pd.DataFrame, datetime_col: str = 'datetime', 
                           group_col: str = 'city') -> Dict[str, Any]:
        """Validate time series data structure"""
        try:
            validation_report = {}
            
            # Check datetime column
            if datetime_col not in df.columns:
                validation_report['datetime_column_missing'] = True
                return validation_report
            
            df_temp = df.copy()
            df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col])
            
            # Check for gaps in time series
            gaps_by_group = {}
            for group in df_temp[group_col].unique():
                group_data = df_temp[df_temp[group_col] == group].sort_values(datetime_col)
                time_diffs = group_data[datetime_col].diff().dropna()
                
                # Expected frequency (assuming hourly data)
                expected_freq = pd.Timedelta(hours=1)
                gaps = time_diffs[time_diffs > expected_freq]
                
                if len(gaps) > 0:
                    gaps_by_group[group] = len(gaps)
            
            validation_report['time_gaps'] = gaps_by_group
            
            # Check date range coverage
            date_range = {
                'start': df_temp[datetime_col].min(),
                'end': df_temp[datetime_col].max(),
                'total_days': (df_temp[datetime_col].max() - df_temp[datetime_col].min()).days
            }
            validation_report['date_range'] = date_range
            
            # Check data frequency
            most_common_diff = df_temp[datetime_col].diff().mode()
            if len(most_common_diff) > 0:
                validation_report['most_common_frequency'] = most_common_diff.iloc[0]
            
            logger.info("Time series validation completed")
            return validation_report
            
        except Exception as e:
            logger.error(f"Error in time series validation: {e}")
            raise

def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all comprehensive features in one function"""
    try:
        processor = DataProcessor()
        
        # Create temporal features
        df_processed = processor.create_temporal_features(df)
        
        # Create lag features
        key_columns = ['temperature', 'humidity', 'aqi', 'pm25', 'pm10']
        available_columns = [col for col in key_columns if col in df.columns]
        df_processed = processor.create_lag_features(df_processed, available_columns, [1, 6, 12, 24])
        
        # Create rolling features
        df_processed = processor.create_rolling_features(df_processed, available_columns, [6, 12, 24, 168])
        
        # Create interaction features
        interaction_pairs = [
            ('temperature', 'humidity'),
            ('wind_speed', 'precipitation'),
            ('pm25', 'humidity'),
            ('temperature', 'pressure')
        ]
        available_pairs = [(f1, f2) for f1, f2 in interaction_pairs 
                          if f1 in df.columns and f2 in df.columns]
        df_processed = processor.create_interaction_features(df_processed, available_pairs)
        
        logger.info(f"Created comprehensive features. Final shape: {df_processed.shape}")
        return df_processed
        
    except Exception as e:
        logger.error(f"Error creating comprehensive features: {e}")
        raise

def main():
    """Test data processing functions"""
    try:
        # Load sample data
        data_path = "data/weather_dataset_cleaned.csv"
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found at {data_path}")
            return
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Validate data quality
        validator = DataValidator()
        quality_report = validator.validate_data_quality(df)
        
        print("Data Quality Report:")
        for key, value in quality_report.items():
            print(f"  {key}: {value}")
        
        # Create comprehensive features
        df_enhanced = create_comprehensive_features(df.head(1000))  # Test with subset
        
        print(f"\nOriginal features: {len(df.columns)}")
        print(f"Enhanced features: {len(df_enhanced.columns)}")
        print(f"New features added: {len(df_enhanced.columns) - len(df.columns)}")
        
    except Exception as e:
        logger.error(f"Data processing test failed: {e}")
        raise

if __name__ == "__main__":
    main()
