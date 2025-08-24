"""
Main Streamlit Application for Weather Prediction AI/ML Project
Features: Rain animation hero, multi-page navigation, dark/light themes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import custom modules
try:
    from utils.cache import CacheManager
    from utils.data_utils import DataProcessor
    from models.model_zoo import WeatherModelZoo
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Weather Prediction AI/ML Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WeatherApp:
    def __init__(self):
        """Initialize the Weather Prediction App"""
        self.load_config()
        self.initialize_session_state()
        self.load_custom_css()
        
    def load_config(self):
        """Load configuration"""
        try:
            config_path = project_root / "config" / "config.yaml"
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            self.config = {}
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'theme' not in st.session_state:
            st.session_state.theme = self.config.get('ui', {}).get('default_theme', 'dark')
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
        
        if 'selected_city' not in st.session_state:
            st.session_state.selected_city = None
        
        if 'selected_target' not in st.session_state:
            st.session_state.selected_target = 'temperature'
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
    
    def load_custom_css(self):
        """Load custom CSS for themes and animations"""
        css_file = project_root / "streamlit" / "assets" / "style.css"
        
        # Default CSS if file doesn't exist
        default_css = """
        <style>
        /* Rain Animation */
        .rain-container {
            position: relative;
            width: 100%;
            height: 100vh;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            overflow: hidden;
        }
        
        .rain {
            position: absolute;
            width: 2px;
            height: 100px;
            background: linear-gradient(transparent, rgba(255,255,255,0.6), transparent);
            animation: fall linear infinite;
        }
        
        @keyframes fall {
            0% { transform: translateY(-100vh); }
            100% { transform: translateY(100vh); }
        }
        
        /* Theme Variables */
        :root {
            --primary-color: #3B82F6;
            --secondary-color: #10B981;
            --accent-color: #F59E0B;
            --danger-color: #EF4444;
            --success-color: #22C55E;
        }
        
        /* Dark Theme */
        .dark-theme {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .dark-theme .stApp {
            background-color: #1a1a1a;
        }
        
        /* Light Theme */
        .light-theme {
            background-color: #ffffff;
            color: #000000;
        }
        
        /* Hero Section */
        .hero-section {
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .hero-subtitle {
            font-size: 1.5rem;
            margin-bottom: 2rem;
            opacity: 0.9;
        }
        
        .get-started-btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 1rem 2rem;
            border: none;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .get-started-btn:hover {
            transform: scale(1.05);
        }
        
        /* Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Animations */
        .fade-in {
            animation: fadeIn 0.8s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Custom buttons */
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        /* Progress bars */
        .stProgress > div > div > div > div {
            background: linear-gradient(45deg, #667eea, #764ba2);
        }
        
        /* Selectbox */
        .stSelectbox > div > div > select {
            border-radius: 10px;
            border: 2px solid #667eea;
        }
        
        /* Metrics */
        .metric-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 2rem 0;
        }
        
        .metric-item {
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin: 0.5rem;
            min-width: 150px;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4ECDC4;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }
        </style>
        """
        
        try:
            if css_file.exists():
                with open(css_file, 'r') as f:
                    custom_css = f.read()
                st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
            else:
                st.markdown(default_css, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(default_css, unsafe_allow_html=True)
    
    def create_rain_animation(self):
        """Create animated rain background"""
        rain_html = """
        <div class="rain-container">
            <div class="rain" style="left: 10%; animation-duration: 1s; animation-delay: 0s;"></div>
            <div class="rain" style="left: 20%; animation-duration: 1.2s; animation-delay: 0.2s;"></div>
            <div class="rain" style="left: 30%; animation-duration: 0.8s; animation-delay: 0.4s;"></div>
            <div class="rain" style="left: 40%; animation-duration: 1.5s; animation-delay: 0.1s;"></div>
            <div class="rain" style="left: 50%; animation-duration: 1.1s; animation-delay: 0.3s;"></div>
            <div class="rain" style="left: 60%; animation-duration: 0.9s; animation-delay: 0.5s;"></div>
            <div class="rain" style="left: 70%; animation-duration: 1.3s; animation-delay: 0.2s;"></div>
            <div class="rain" style="left: 80%; animation-duration: 1.0s; animation-delay: 0.4s;"></div>
            <div class="rain" style="left: 90%; animation-duration: 1.4s; animation-delay: 0.1s;"></div>
        </div>
        """
        return rain_html
    
    def render_hero_section(self):
        """Render the hero section with rain animation"""
        # Rain animation background
        st.markdown(self.create_rain_animation(), unsafe_allow_html=True)
        
        # Hero content
        hero_html = """
        <div class="hero-section fade-in">
            <h1 class="hero-title">🌦️ Weather Prediction AI/ML</h1>
            <p class="hero-subtitle">
                Advanced Machine Learning for Weather & Air Quality Forecasting<br>
                Covering 120+ Indian Cities with 35+ Features
            </p>
            <div style="margin-top: 2rem;">
                <div class="metric-container">
                    <div class="metric-item">
                        <div class="metric-value">120+</div>
                        <div class="metric-label">Indian Cities</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">35+</div>
                        <div class="metric-label">Weather Features</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">13</div>
                        <div class="metric-label">ML Models</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">7 Days</div>
                        <div class="metric-label">Forecast Range</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(hero_html, unsafe_allow_html=True)
        
        # Get Started Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Get Started", key="get_started", help="Start exploring weather predictions"):
                st.session_state.current_page = 'dashboard'
                st.rerun()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.markdown("### 🌦️ Weather AI Dashboard")
            
            # Theme toggle
            theme_options = ['dark', 'light', 'monsoon', 'summer', 'winter']
            current_theme = st.selectbox(
                "🎨 Theme",
                theme_options,
                index=theme_options.index(st.session_state.theme)
            )
            
            if current_theme != st.session_state.theme:
                st.session_state.theme = current_theme
                st.rerun()
            
            st.markdown("---")
            
            # Navigation
            pages = {
                'home': '🏠 Home',
                'dashboard': '📊 Dashboard', 
                'city_analysis': '🏙️ City Analysis',
                'model_comparison': '🤖 Model Comparison',
                'predictions': '🔮 Predictions',
                'data_explorer': '📈 Data Explorer',
                'settings': '⚙️ Settings'
            }
            
            for page_key, page_name in pages.items():
                if st.button(page_name, key=f"nav_{page_key}"):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            st.markdown("---")
            
            # Quick stats
            st.markdown("### 📊 Quick Stats")
            
            # Check if data exists
            data_path = project_root / "data" / "weather_dataset.csv"
            if data_path.exists():
                try:
                    # Load sample for stats
                    df_sample = pd.read_csv(data_path, nrows=1000)
                    
                    st.metric("Cities Available", df_sample['city'].nunique())
                    st.metric("Data Points", f"{len(df_sample)}+")
                    st.metric("Features", len(df_sample.columns))
                    
                    # Latest data timestamp
                    if 'datetime' in df_sample.columns:
                        latest_date = pd.to_datetime(df_sample['datetime']).max()
                        st.metric("Latest Data", latest_date.strftime('%Y-%m-%d'))
                    
                except Exception as e:
                    st.error(f"Error loading data stats: {e}")
            else:
                st.warning("No data available. Please generate dataset first.")
            
            st.markdown("---")
            
            # System status
            st.markdown("### 🔧 System Status")
            
            # Check various components
            status_items = [
                ("Dataset", data_path.exists()),
                ("Models", (project_root / "models" / "trained").exists()),
                ("Cache", (project_root / "cache").exists()),
                ("EDA Results", (project_root / "eda" / "plots").exists())
            ]
            
            for item, status in status_items:
                status_icon = "✅" if status else "❌"
                st.markdown(f"{status_icon} {item}")
    
    def load_data(self):
        """Load weather dataset"""
        try:
            data_path = project_root / "data" / "weather_dataset_cleaned.csv"
            if not data_path.exists():
                data_path = project_root / "data" / "weather_dataset.csv"
            
            if data_path.exists():
                df = pd.read_csv(data_path)
                df['datetime'] = pd.to_datetime(df['datetime'])
                st.session_state.data_loaded = True
                return df
            else:
                st.error("Dataset not found. Please generate the dataset first.")
                return None
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.title("📊 Weather Prediction Dashboard")
        
        # Load data
        df = self.load_data()
        if df is None:
            st.warning("Please generate the dataset first by running the data generation script.")
            
            if st.button("🔄 Generate Dataset"):
                with st.spinner("Generating synthetic dataset... This may take a few minutes."):
                    try:
                        # Run data generation
                        import subprocess
                        result = subprocess.run([
                            "python", str(project_root / "data" / "generate_dataset.py")
                        ], capture_output=True, text=True, cwd=str(project_root))
                        
                        if result.returncode == 0:
                            st.success("Dataset generated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error generating dataset: {result.stderr}")
                    except Exception as e:
                        st.error(f"Error running data generation: {e}")
            return
        
        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records",
                f"{len(df):,}",
                delta=f"+{len(df) - len(df)*0.95:.0f} (5%)"
            )
        
        with col2:
            st.metric(
                "Cities Covered",
                df['city'].nunique(),
                delta=f"+{df['city'].nunique() - 100}"
            )
        
        with col3:
            avg_temp = df['temperature'].mean()
            st.metric(
                "Avg Temperature",
                f"{avg_temp:.1f}°C",
                delta=f"{avg_temp - 25:.1f}°C"
            )
        
        with col4:
            avg_aqi = df['aqi'].mean()
            st.metric(
                "Avg AQI",
                f"{avg_aqi:.0f}",
                delta=f"{avg_aqi - 100:.0f}",
                delta_color="inverse"
            )
        
        # Quick visualizations
        st.markdown("### 📈 Overview Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature trend by city (top 10)
            top_cities = df['city'].value_counts().head(10).index
            city_temp = df[df['city'].isin(top_cities)].groupby('city')['temperature'].mean().sort_values(ascending=False)
            
            fig_temp = px.bar(
                x=city_temp.values,
                y=city_temp.index,
                orientation='h',
                title="Average Temperature by City (Top 10)",
                labels={'x': 'Temperature (°C)', 'y': 'City'},
                color=city_temp.values,
                color_continuous_scale='RdYlBu_r'
            )
            fig_temp.update_layout(height=400)
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            # AQI distribution
            fig_aqi = px.histogram(
                df.sample(1000),  # Sample for performance
                x='aqi',
                nbins=30,
                title="AQI Distribution",
                labels={'x': 'AQI', 'y': 'Frequency'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig_aqi.update_layout(height=400)
            st.plotly_chart(fig_aqi, use_container_width=True)
        
        # Time series preview
        st.markdown("### 📅 Time Series Preview")
        
        # Select a sample city for time series
        sample_city = df['city'].iloc[0]
        city_data = df[df['city'] == sample_city].sort_values('datetime').tail(168)  # Last week
        
        fig_ts = go.Figure()
        
        fig_ts.add_trace(go.Scatter(
            x=city_data['datetime'],
            y=city_data['temperature'],
            mode='lines',
            name='Temperature',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=city_data['datetime'],
            y=city_data['humidity'],
            mode='lines',
            name='Humidity',
            yaxis='y2',
            line=dict(color='#4ECDC4', width=2)
        ))
        
        fig_ts.update_layout(
            title=f"Weather Trends - {sample_city} (Last Week)",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            yaxis2=dict(
                title="Humidity (%)",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Action buttons
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏙️ Analyze Cities"):
                st.session_state.current_page = 'city_analysis'
                st.rerun()
        
        with col2:
            if st.button("🤖 Compare Models"):
                st.session_state.current_page = 'model_comparison'
                st.rerun()
        
        with col3:
            if st.button("🔮 Make Predictions"):
                st.session_state.current_page = 'predictions'
                st.rerun()
        
        with col4:
            if st.button("📊 Explore Data"):
                st.session_state.current_page = 'data_explorer'
                st.rerun()
    
    def render_placeholder_page(self, page_name):
        """Render placeholder for pages under development"""
        st.title(f"🚧 {page_name.replace('_', ' ').title()}")
        
        st.info(f"The {page_name.replace('_', ' ').title()} page is under development.")
        
        # Show some relevant content based on page
        if page_name == 'city_analysis':
            st.markdown("""
            ### Coming Soon: City Analysis
            - Compare weather patterns across cities
            - Interactive city selection
            - Detailed city-specific insights
            - Geographic visualizations
            """)
        
        elif page_name == 'model_comparison':
            st.markdown("""
            ### Coming Soon: Model Comparison
            - Performance metrics for all models
            - Interactive model selection
            - Feature importance analysis
            - Cross-validation results
            """)
        
        elif page_name == 'predictions':
            st.markdown("""
            ### Coming Soon: Predictions
            - 7-day weather forecasts
            - Confidence intervals
            - Multiple target variables
            - Export predictions
            """)
        
        elif page_name == 'data_explorer':
            st.markdown("""
            ### Coming Soon: Data Explorer
            - Interactive data filtering
            - Custom visualizations
            - Statistical analysis
            - Data export options
            """)
        
        elif page_name == 'settings':
            st.markdown("""
            ### Coming Soon: Settings
            - Theme customization
            - Model parameters
            - Data refresh settings
            - Export preferences
            """)
        
        # Back to dashboard button
        if st.button("← Back to Dashboard"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        self.render_sidebar()
        
        # Main content based on current page
        if st.session_state.current_page == 'home':
            self.render_hero_section()
        elif st.session_state.current_page == 'dashboard':
            self.render_dashboard()
        else:
            self.render_placeholder_page(st.session_state.current_page)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; opacity: 0.7; padding: 1rem;'>
                🌦️ Weather Prediction AI/ML Dashboard | Built with Streamlit & Python<br>
                Hackathon Project - Advanced Weather & AQI Forecasting
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main function"""
    try:
        app = WeatherApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
