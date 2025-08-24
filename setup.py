#!/usr/bin/env python3
"""
Quick Setup Script for Weather Prediction AI/ML Project
Automates the entire setup and execution process
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherProjectSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.steps_completed = 0
        self.total_steps = 6
        
    def print_banner(self):
        """Print project banner"""
        banner = """
        ╔══════════════════════════════════════════════════════════════╗
        ║                                                              ║
        ║           🌦️  WEATHER PREDICTION AI/ML PROJECT  🌦️           ║
        ║                                                              ║
        ║              🏆 HACKATHON-WINNING SOLUTION 🏆                ║
        ║                                                              ║
        ║  Features: 120+ Cities | 35+ Features | 13 ML Models        ║
        ║           Interactive Dashboard | Rain Animations            ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print("🚀 Starting automated setup process...\n")
    
    def print_progress(self, step_name, status="running"):
        """Print progress with emojis"""
        if status == "running":
            print(f"⏳ Step {self.steps_completed + 1}/{self.total_steps}: {step_name}...")
        elif status == "completed":
            self.steps_completed += 1
            print(f"✅ Step {self.steps_completed}/{self.total_steps}: {step_name} - COMPLETED")
        elif status == "error":
            print(f"❌ Step {self.steps_completed + 1}/{self.total_steps}: {step_name} - FAILED")
    
    def run_command(self, command, description, cwd=None):
        """Run a command with error handling"""
        try:
            self.print_progress(description, "running")
            
            if cwd is None:
                cwd = self.project_root
            
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.print_progress(description, "completed")
                return True
            else:
                self.print_progress(description, "error")
                logger.error(f"Command failed: {command}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.print_progress(description, "error")
            logger.error(f"Command timed out: {command}")
            return False
        except Exception as e:
            self.print_progress(description, "error")
            logger.error(f"Error running command: {e}")
            return False
    
    def check_python_version(self):
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ is required. Current version:", sys.version)
            return False
        print(f"✅ Python version check passed: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        return self.run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing Python dependencies"
        )
    
    def generate_dataset(self):
        """Generate synthetic dataset"""
        return self.run_command(
            f"{sys.executable} data/generate_dataset.py",
            "Generating synthetic weather dataset (120+ cities, 35+ features)"
        )
    
    def run_eda_analysis(self):
        """Run EDA analysis"""
        return self.run_command(
            f"{sys.executable} eda/eda_analysis.py",
            "Running comprehensive EDA analysis"
        )
    
    def train_models(self):
        """Train ML models"""
        return self.run_command(
            f"{sys.executable} models/model_zoo.py",
            "Training 13 ML models with hyperparameter tuning"
        )
    
    def launch_dashboard(self):
        """Launch Streamlit dashboard"""
        print("🎨 Launching interactive Streamlit dashboard...")
        print("📱 Dashboard will open at: http://localhost:8000")
        print("🌦️ Features: Rain animations, multiple themes, interactive charts")
        print("\n🎯 Press Ctrl+C to stop the dashboard when done\n")
        
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "streamlit/app.py", "--server.port", "8000"
            ], cwd=self.project_root)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped. Thank you for using Weather Prediction AI!")
    
    def print_success_message(self):
        """Print success message with instructions"""
        success_message = """
        ╔══════════════════════════════════════════════════════════════╗
        ║                                                              ║
        ║                    🎉 SETUP COMPLETED! 🎉                    ║
        ║                                                              ║
        ║              🏆 PROJECT READY FOR HACKATHON! 🏆              ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        
        📊 WHAT'S BEEN CREATED:
        ✅ Synthetic dataset: 1M+ records, 125 cities, 35+ features
        ✅ EDA analysis: Comprehensive visualizations and insights
        ✅ ML models: 13 trained models with performance metrics
        ✅ Interactive dashboard: Beautiful UI with rain animations
        
        🚀 QUICK COMMANDS:
        • Launch dashboard: streamlit run streamlit/app.py --server.port 8000
        • Re-generate data: python data/generate_dataset.py
        • Re-run EDA: python eda/eda_analysis.py
        • Re-train models: python models/model_zoo.py
        
        📁 KEY FILES CREATED:
        • data/weather_dataset.csv (Main dataset)
        • eda/plots/ (EDA visualizations)
        • models/results/ (Model performance results)
        • models/trained/ (Saved ML models)
        
        🌐 DASHBOARD FEATURES:
        • 🌧️ Animated rain background
        • 📊 Interactive charts and metrics
        • 🎨 Multiple themes (dark/light/seasonal)
        • 📱 Mobile-responsive design
        • 🏙️ City-wise analysis
        • 🤖 Model comparison
        • 🔮 Weather predictions
        
        🏆 HACKATHON READY FEATURES:
        • Comprehensive 120+ Indian cities coverage
        • Advanced ML pipeline with 13 models
        • Beautiful interactive dashboard
        • Production-quality code
        • Complete documentation
        
        """
        print(success_message)
    
    def run_setup(self):
        """Run complete setup process"""
        start_time = time.time()
        
        # Print banner
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Run setup steps
        steps = [
            (self.install_dependencies, "Dependencies installation"),
            (self.generate_dataset, "Dataset generation"),
            (self.run_eda_analysis, "EDA analysis"),
            (self.train_models, "Model training"),
        ]
        
        for step_func, step_name in steps:
            if not step_func():
                print(f"\n❌ Setup failed at: {step_name}")
                print("🔧 Please check the error messages above and try again.")
                return False
            print()  # Add spacing between steps
        
        # Calculate total time
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print(f"⏱️ Total setup time: {minutes}m {seconds}s")
        
        # Print success message
        self.print_success_message()
        
        # Ask if user wants to launch dashboard
        try:
            launch = input("🚀 Launch dashboard now? (y/n): ").lower().strip()
            if launch in ['y', 'yes', '']:
                self.launch_dashboard()
        except KeyboardInterrupt:
            print("\n👋 Setup completed successfully!")
        
        return True

def main():
    """Main function"""
    try:
        setup = WeatherProjectSetup()
        success = setup.run_setup()
        
        if success:
            print("\n🎉 Weather Prediction AI/ML project is ready!")
            print("🏆 Good luck with your hackathon!")
        else:
            print("\n❌ Setup encountered errors. Please check the logs above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
