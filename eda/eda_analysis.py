"""
Comprehensive Exploratory Data Analysis (EDA) for Weather Prediction Dataset
Includes data cleaning, visualization, and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import logging
from typing import Dict, List, Tuple, Optional
import yaml
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Setup
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherEDA:
    def __init__(self, data_path: str, config_path: str = "config/config.yaml"):
        """Initialize EDA with dataset and configuration"""
        try:
            # Load configuration
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Load dataset
            if data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                self.df = pd.read_parquet(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet.")
            
            # Convert datetime column
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            
            # Create output directory for plots
            self.output_dir = "eda/plots"
            os.makedirs(self.output_dir, exist_ok=True)
            
            logger.info(f"Loaded dataset with {len(self.df)} records and {len(self.df.columns)} features")
            
        except Exception as e:
            logger.error(f"Error initializing EDA: {e}")
            raise
    
    def basic_info(self) -> Dict:
        """Generate basic information about the dataset"""
        try:
            info = {
                'shape': self.df.shape,
                'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'date_range': f"{self.df['datetime'].min()} to {self.df['datetime'].max()}",
                'cities_count': self.df['city'].nunique(),
                'missing_values': self.df.isnull().sum().sum(),
                'duplicate_rows': self.df.duplicated().sum(),
                'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
            }
            
            logger.info("Basic dataset information generated")
            return info
            
        except Exception as e:
            logger.error(f"Error generating basic info: {e}")
            raise
    
    def missing_value_analysis(self) -> pd.DataFrame:
        """Analyze missing values in the dataset"""
        try:
            missing_data = pd.DataFrame({
                'Column': self.df.columns,
                'Missing_Count': self.df.isnull().sum(),
                'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
            })
            
            missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
            
            # Create visualization
            if not missing_data.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Missing count plot
                sns.barplot(data=missing_data.head(10), y='Column', x='Missing_Count', ax=ax1)
                ax1.set_title('Top 10 Columns with Missing Values (Count)')
                ax1.set_xlabel('Missing Count')
                
                # Missing percentage plot
                sns.barplot(data=missing_data.head(10), y='Column', x='Missing_Percentage', ax=ax2)
                ax2.set_title('Top 10 Columns with Missing Values (%)')
                ax2.set_xlabel('Missing Percentage')
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/missing_values_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("Missing value analysis completed")
            return missing_data
            
        except Exception as e:
            logger.error(f"Error in missing value analysis: {e}")
            raise
    
    def data_distribution_analysis(self) -> None:
        """Analyze distribution of key variables"""
        try:
            # Select key numeric columns
            key_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation', 
                          'aqi', 'pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
            
            # Filter columns that exist in the dataset
            available_columns = [col for col in key_columns if col in self.df.columns]
            
            # Create distribution plots
            n_cols = 3
            n_rows = (len(available_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(available_columns):
                if i < len(axes):
                    # Histogram with KDE
                    sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col.title()}')
                    axes[i].set_xlabel(col.title())
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(available_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/data_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Data distribution analysis completed")
            
        except Exception as e:
            logger.error(f"Error in data distribution analysis: {e}")
            raise
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Analyze correlations between variables"""
        try:
            # Select numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlation_matrix = self.df[numeric_cols].corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(20, 16))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
            plt.title('Correlation Matrix of Weather and AQI Features', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'Feature_1': correlation_matrix.columns[i],
                            'Feature_2': correlation_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            
            logger.info("Correlation analysis completed")
            return high_corr_df
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            raise
    
    def temporal_analysis(self) -> None:
        """Analyze temporal patterns in the data"""
        try:
            # Set datetime as index for time series analysis
            df_temp = self.df.set_index('datetime')
            
            # Monthly patterns
            monthly_stats = df_temp.groupby(df_temp.index.month).agg({
                'temperature': ['mean', 'std'],
                'humidity': ['mean', 'std'],
                'aqi': ['mean', 'std'],
                'pm25': ['mean', 'std']
            }).round(2)
            
            # Create temporal plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Monthly temperature pattern
            monthly_temp = df_temp.groupby(df_temp.index.month)['temperature'].mean()
            axes[0, 0].plot(monthly_temp.index, monthly_temp.values, marker='o', linewidth=2)
            axes[0, 0].set_title('Average Temperature by Month')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Temperature (°C)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Monthly AQI pattern
            monthly_aqi = df_temp.groupby(df_temp.index.month)['aqi'].mean()
            axes[0, 1].plot(monthly_aqi.index, monthly_aqi.values, marker='s', color='red', linewidth=2)
            axes[0, 1].set_title('Average AQI by Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('AQI')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Hourly temperature pattern
            hourly_temp = df_temp.groupby(df_temp.index.hour)['temperature'].mean()
            axes[1, 0].plot(hourly_temp.index, hourly_temp.values, marker='^', color='orange', linewidth=2)
            axes[1, 0].set_title('Average Temperature by Hour of Day')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Day of week pattern for AQI
            dow_aqi = df_temp.groupby(df_temp.index.dayofweek)['aqi'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[1, 1].bar(days, dow_aqi.values, color='purple', alpha=0.7)
            axes[1, 1].set_title('Average AQI by Day of Week')
            axes[1, 1].set_xlabel('Day of Week')
            axes[1, 1].set_ylabel('AQI')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/temporal_patterns.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Temporal analysis completed")
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            raise
    
    def city_comparison_analysis(self) -> None:
        """Compare weather patterns across different cities"""
        try:
            # Select top 10 cities by data availability
            city_counts = self.df['city'].value_counts().head(10)
            top_cities = city_counts.index.tolist()
            
            # Filter data for top cities
            city_data = self.df[self.df['city'].isin(top_cities)]
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Average temperature by city
            city_temp = city_data.groupby('city')['temperature'].mean().sort_values(ascending=False)
            axes[0, 0].barh(range(len(city_temp)), city_temp.values, color='skyblue')
            axes[0, 0].set_yticks(range(len(city_temp)))
            axes[0, 0].set_yticklabels(city_temp.index)
            axes[0, 0].set_title('Average Temperature by City')
            axes[0, 0].set_xlabel('Temperature (°C)')
            
            # Average AQI by city
            city_aqi = city_data.groupby('city')['aqi'].mean().sort_values(ascending=False)
            axes[0, 1].barh(range(len(city_aqi)), city_aqi.values, color='red', alpha=0.7)
            axes[0, 1].set_yticks(range(len(city_aqi)))
            axes[0, 1].set_yticklabels(city_aqi.index)
            axes[0, 1].set_title('Average AQI by City')
            axes[0, 1].set_xlabel('AQI')
            
            # Humidity distribution by city (box plot)
            sns.boxplot(data=city_data, y='city', x='humidity', ax=axes[1, 0])
            axes[1, 0].set_title('Humidity Distribution by City')
            axes[1, 0].set_xlabel('Humidity (%)')
            
            # PM2.5 distribution by city (box plot)
            sns.boxplot(data=city_data, y='city', x='pm25', ax=axes[1, 1])
            axes[1, 1].set_title('PM2.5 Distribution by City')
            axes[1, 1].set_xlabel('PM2.5 (μg/m³)')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/city_comparisons.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("City comparison analysis completed")
            
        except Exception as e:
            logger.error(f"Error in city comparison analysis: {e}")
            raise
    
    def outlier_detection(self) -> Dict:
        """Detect outliers in key variables"""
        try:
            key_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm25', 'pm10']
            available_columns = [col for col in key_columns if col in self.df.columns]
            
            outlier_info = {}
            
            # Create outlier plots
            n_cols = 3
            n_rows = (len(available_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(available_columns):
                if i < len(axes):
                    # Calculate IQR method outliers
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_percentage = (outlier_count / len(self.df)) * 100
                    
                    outlier_info[col] = {
                        'count': outlier_count,
                        'percentage': outlier_percentage,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    
                    # Box plot
                    sns.boxplot(data=self.df, y=col, ax=axes[i])
                    axes[i].set_title(f'{col.title()} - Outliers: {outlier_count} ({outlier_percentage:.1f}%)')
            
            # Hide unused subplots
            for i in range(len(available_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/outlier_detection.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Outlier detection completed")
            return outlier_info
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            raise
    
    def feature_importance_analysis(self) -> None:
        """Analyze feature importance using correlation with target variables"""
        try:
            target_variables = ['temperature', 'aqi', 'pm25']
            available_targets = [col for col in target_variables if col in self.df.columns]
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['datetime'] and col not in available_targets]
            
            fig, axes = plt.subplots(1, len(available_targets), figsize=(6*len(available_targets), 8))
            if len(available_targets) == 1:
                axes = [axes]
            
            for i, target in enumerate(available_targets):
                # Calculate correlations
                correlations = self.df[feature_cols + [target]].corr()[target].drop(target)
                correlations = correlations.abs().sort_values(ascending=True)
                
                # Plot top 15 features
                top_features = correlations.tail(15)
                axes[i].barh(range(len(top_features)), top_features.values, color='lightcoral')
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features.index)
                axes[i].set_title(f'Feature Importance for {target.title()}')
                axes[i].set_xlabel('Absolute Correlation')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature importance analysis completed")
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            raise
    
    def data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        try:
            report = {}
            
            # Basic statistics
            report['basic_info'] = self.basic_info()
            
            # Missing values
            report['missing_values'] = self.missing_value_analysis().to_dict('records')
            
            # Outliers
            report['outliers'] = self.outlier_detection()
            
            # Data types
            report['data_types'] = self.df.dtypes.to_dict()
            
            # Summary statistics
            numeric_summary = self.df.describe().to_dict()
            report['summary_statistics'] = numeric_summary
            
            # Unique values for categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            report['categorical_info'] = {}
            for col in categorical_cols:
                report['categorical_info'][col] = {
                    'unique_count': self.df[col].nunique(),
                    'top_values': self.df[col].value_counts().head().to_dict()
                }
            
            logger.info("Data quality report generated")
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset by handling missing values and outliers"""
        try:
            df_cleaned = self.df.copy()
            
            # Handle missing values
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            
            # Fill missing values with median for numeric columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled {col} missing values with median: {median_val:.2f}")
            
            # Handle categorical missing values
            categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    mode_val = df_cleaned[col].mode()[0]
                    df_cleaned[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled {col} missing values with mode: {mode_val}")
            
            # Handle extreme outliers (beyond 3 standard deviations)
            for col in ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm25', 'pm10']:
                if col in df_cleaned.columns:
                    mean_val = df_cleaned[col].mean()
                    std_val = df_cleaned[col].std()
                    lower_limit = mean_val - 3 * std_val
                    upper_limit = mean_val + 3 * std_val
                    
                    # Cap extreme values
                    outliers_count = len(df_cleaned[(df_cleaned[col] < lower_limit) | (df_cleaned[col] > upper_limit)])
                    df_cleaned[col] = np.clip(df_cleaned[col], lower_limit, upper_limit)
                    
                    if outliers_count > 0:
                        logger.info(f"Capped {outliers_count} extreme outliers in {col}")
            
            # Add lag features for time series
            df_cleaned = df_cleaned.sort_values(['city', 'datetime'])
            
            for col in ['temperature', 'humidity', 'aqi', 'pm25']:
                if col in df_cleaned.columns:
                    df_cleaned[f'{col}_lag_1h'] = df_cleaned.groupby('city')[col].shift(1)
                    df_cleaned[f'{col}_lag_24h'] = df_cleaned.groupby('city')[col].shift(24)
            
            # Add rolling averages
            for col in ['temperature', 'aqi', 'pm25']:
                if col in df_cleaned.columns:
                    df_cleaned[f'{col}_rolling_24h'] = df_cleaned.groupby('city')[col].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
                    df_cleaned[f'{col}_rolling_7d'] = df_cleaned.groupby('city')[col].rolling(window=168, min_periods=1).mean().reset_index(0, drop=True)  # 7 days * 24 hours
            
            logger.info(f"Data cleaning completed. Shape: {df_cleaned.shape}")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise
    
    def generate_eda_report(self) -> str:
        """Generate comprehensive EDA report"""
        try:
            logger.info("Starting comprehensive EDA analysis...")
            
            # Run all analyses
            basic_info = self.basic_info()
            missing_analysis = self.missing_value_analysis()
            self.data_distribution_analysis()
            correlation_analysis = self.correlation_analysis()
            self.temporal_analysis()
            self.city_comparison_analysis()
            outlier_info = self.outlier_detection()
            self.feature_importance_analysis()
            quality_report = self.data_quality_report()
            
            # Generate summary report
            report_lines = [
                "="*60,
                "WEATHER PREDICTION DATASET - EDA REPORT",
                "="*60,
                f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "DATASET OVERVIEW:",
                f"• Total Records: {basic_info['shape'][0]:,}",
                f"• Total Features: {basic_info['shape'][1]}",
                f"• Memory Usage: {basic_info['memory_usage']}",
                f"• Date Range: {basic_info['date_range']}",
                f"• Cities Covered: {basic_info['cities_count']}",
                "",
                "DATA QUALITY:",
                f"• Missing Values: {basic_info['missing_values']:,} ({(basic_info['missing_values']/(basic_info['shape'][0]*basic_info['shape'][1]))*100:.2f}%)",
                f"• Duplicate Rows: {basic_info['duplicate_rows']:,}",
                f"• Numeric Columns: {basic_info['numeric_columns']}",
                f"• Categorical Columns: {basic_info['categorical_columns']}",
                "",
                "KEY FINDINGS:",
                f"• High Correlation Pairs Found: {len(correlation_analysis)}",
                f"• Outliers Detected in Multiple Variables",
                f"• Clear Temporal Patterns Observed",
                f"• Significant City-wise Variations Present",
                "",
                "GENERATED VISUALIZATIONS:",
                "• Data Distributions",
                "• Correlation Matrix",
                "• Temporal Patterns",
                "• City Comparisons",
                "• Outlier Detection",
                "• Feature Importance",
                "• Missing Value Analysis",
                "",
                f"All plots saved in: {self.output_dir}/",
                "="*60
            ]
            
            report_text = "\n".join(report_lines)
            
            # Save report
            report_path = f"{self.output_dir}/eda_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)
            
            logger.info(f"EDA report generated: {report_path}")
            print(report_text)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating EDA report: {e}")
            raise

def main():
    """Main function to run EDA analysis"""
    try:
        # Check if dataset exists
        data_path = "data/weather_dataset.csv"
        if not os.path.exists(data_path):
            logger.error(f"Dataset not found at {data_path}. Please run data generation first.")
            return
        
        # Initialize EDA
        eda = WeatherEDA(data_path)
        
        # Generate comprehensive report
        report_path = eda.generate_eda_report()
        
        # Clean data and save
        cleaned_data = eda.clean_data()
        cleaned_path = "data/weather_dataset_cleaned.csv"
        cleaned_data.to_csv(cleaned_path, index=False)
        logger.info(f"Cleaned dataset saved: {cleaned_path}")
        
        print(f"\nEDA Analysis Complete!")
        print(f"Report: {report_path}")
        print(f"Cleaned Data: {cleaned_path}")
        print(f"Plots Directory: {eda.output_dir}")
        
    except Exception as e:
        logger.error(f"EDA analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
