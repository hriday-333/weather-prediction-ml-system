"""
Caching Utilities for Weather Prediction Project
Provides persistent caching for models, data, and results
"""

import os
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import hashlib
import yaml
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CacheManager:
    """Comprehensive cache management system"""
    
    def __init__(self, cache_dir: str = "cache", config_path: str = "config/config.yaml"):
        """Initialize cache manager"""
        try:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            self.data_cache_dir = self.cache_dir / "data"
            self.model_cache_dir = self.cache_dir / "models"
            self.results_cache_dir = self.cache_dir / "results"
            self.temp_cache_dir = self.cache_dir / "temp"
            
            for dir_path in [self.data_cache_dir, self.model_cache_dir, 
                           self.results_cache_dir, self.temp_cache_dir]:
                dir_path.mkdir(exist_ok=True)
            
            # Load configuration
            try:
                with open(config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
                self.cache_enabled = self.config.get('cache', {}).get('enabled', True)
                self.default_expiry_hours = self.config.get('cache', {}).get('model_cache_hours', 24)
            except FileNotFoundError:
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                self.cache_enabled = True
                self.default_expiry_hours = 24
            
            # Cache metadata
            self.metadata_file = self.cache_dir / "cache_metadata.json"
            self.metadata = self._load_metadata()
            
            logger.info(f"Cache manager initialized. Cache dir: {self.cache_dir}")
            
        except Exception as e:
            logger.error(f"Error initializing cache manager: {e}")
            raise
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate unique cache key from arguments"""
        try:
            # Create a string representation of all arguments
            key_data = str(args) + str(sorted(kwargs.items()))
            # Generate hash
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return str(hash(str(args) + str(kwargs)))
    
    def _is_expired(self, cache_key: str, expiry_hours: Optional[float] = None) -> bool:
        """Check if cache entry is expired"""
        try:
            if cache_key not in self.metadata:
                return True
            
            if expiry_hours is None:
                expiry_hours = self.default_expiry_hours
            
            cached_time = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
            expiry_time = cached_time + timedelta(hours=expiry_hours)
            
            return datetime.now() > expiry_time
            
        except Exception as e:
            logger.error(f"Error checking cache expiry: {e}")
            return True
    
    def cache_data(self, data: Union[pd.DataFrame, np.ndarray, Dict, List], 
                   cache_key: str, expiry_hours: Optional[float] = None) -> bool:
        """Cache data with automatic format detection"""
        try:
            if not self.cache_enabled:
                return False
            
            cache_path = self.data_cache_dir / f"{cache_key}.pkl"
            
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'type': 'data',
                'timestamp': datetime.now().isoformat(),
                'file_path': str(cache_path),
                'data_type': type(data).__name__,
                'size_bytes': cache_path.stat().st_size if cache_path.exists() else 0
            }
            
            if isinstance(data, pd.DataFrame):
                self.metadata[cache_key]['shape'] = data.shape
                self.metadata[cache_key]['columns'] = list(data.columns)
            elif isinstance(data, np.ndarray):
                self.metadata[cache_key]['shape'] = data.shape
            
            self._save_metadata()
            logger.info(f"Cached data with key: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            return False
    
    def load_cached_data(self, cache_key: str, expiry_hours: Optional[float] = None) -> Optional[Any]:
        """Load cached data if available and not expired"""
        try:
            if not self.cache_enabled:
                return None
            
            if self._is_expired(cache_key, expiry_hours):
                logger.info(f"Cache expired for key: {cache_key}")
                return None
            
            cache_path = self.data_cache_dir / f"{cache_key}.pkl"
            
            if not cache_path.exists():
                logger.info(f"Cache file not found: {cache_path}")
                return None
            
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded cached data with key: {cache_key}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return None
    
    def cache_model(self, model: Any, model_name: str, target: str, 
                   metadata: Optional[Dict] = None, expiry_hours: Optional[float] = None) -> bool:
        """Cache trained model"""
        try:
            if not self.cache_enabled:
                return False
            
            cache_key = f"{target}_{model_name}"
            cache_path = self.model_cache_dir / f"{cache_key}.joblib"
            
            # Save model
            joblib.dump(model, cache_path)
            
            # Update metadata
            model_metadata = {
                'type': 'model',
                'timestamp': datetime.now().isoformat(),
                'file_path': str(cache_path),
                'model_name': model_name,
                'target': target,
                'size_bytes': cache_path.stat().st_size if cache_path.exists() else 0
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            self.metadata[cache_key] = model_metadata
            self._save_metadata()
            
            logger.info(f"Cached model: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching model: {e}")
            return False
    
    def load_cached_model(self, model_name: str, target: str, 
                         expiry_hours: Optional[float] = None) -> Optional[Any]:
        """Load cached model if available and not expired"""
        try:
            if not self.cache_enabled:
                return None
            
            cache_key = f"{target}_{model_name}"
            
            if self._is_expired(cache_key, expiry_hours):
                logger.info(f"Model cache expired for: {cache_key}")
                return None
            
            cache_path = self.model_cache_dir / f"{cache_key}.joblib"
            
            if not cache_path.exists():
                logger.info(f"Model cache file not found: {cache_path}")
                return None
            
            model = joblib.load(cache_path)
            logger.info(f"Loaded cached model: {cache_key}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading cached model: {e}")
            return None
    
    def cache_results(self, results: Dict, results_key: str, 
                     expiry_hours: Optional[float] = None) -> bool:
        """Cache analysis results"""
        try:
            if not self.cache_enabled:
                return False
            
            cache_path = self.results_cache_dir / f"{results_key}.json"
            
            # Convert numpy arrays and other non-serializable objects
            serializable_results = self._make_serializable(results)
            
            # Save results
            with open(cache_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Update metadata
            self.metadata[results_key] = {
                'type': 'results',
                'timestamp': datetime.now().isoformat(),
                'file_path': str(cache_path),
                'size_bytes': cache_path.stat().st_size if cache_path.exists() else 0
            }
            
            self._save_metadata()
            logger.info(f"Cached results with key: {results_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching results: {e}")
            return False
    
    def load_cached_results(self, results_key: str, 
                           expiry_hours: Optional[float] = None) -> Optional[Dict]:
        """Load cached results if available and not expired"""
        try:
            if not self.cache_enabled:
                return None
            
            if self._is_expired(results_key, expiry_hours):
                logger.info(f"Results cache expired for: {results_key}")
                return None
            
            cache_path = self.results_cache_dir / f"{results_key}.json"
            
            if not cache_path.exists():
                logger.info(f"Results cache file not found: {cache_path}")
                return None
            
            with open(cache_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded cached results: {results_key}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading cached results: {e}")
            return None
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format"""
        try:
            if isinstance(obj, dict):
                return {key: self._make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self._make_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return self._make_serializable(obj.__dict__)
            else:
                return obj
        except Exception as e:
            logger.error(f"Error making object serializable: {e}")
            return str(obj)
    
    def get_cache_info(self) -> Dict:
        """Get comprehensive cache information"""
        try:
            cache_info = {
                'cache_enabled': self.cache_enabled,
                'cache_directory': str(self.cache_dir),
                'total_entries': len(self.metadata),
                'cache_size_mb': self._get_cache_size() / (1024 * 1024),
                'entries_by_type': {},
                'expired_entries': [],
                'recent_entries': []
            }
            
            # Analyze entries by type
            for key, meta in self.metadata.items():
                entry_type = meta.get('type', 'unknown')
                if entry_type not in cache_info['entries_by_type']:
                    cache_info['entries_by_type'][entry_type] = 0
                cache_info['entries_by_type'][entry_type] += 1
                
                # Check for expired entries
                if self._is_expired(key):
                    cache_info['expired_entries'].append(key)
                
                # Recent entries (last 24 hours)
                try:
                    entry_time = datetime.fromisoformat(meta['timestamp'])
                    if datetime.now() - entry_time < timedelta(hours=24):
                        cache_info['recent_entries'].append({
                            'key': key,
                            'type': entry_type,
                            'timestamp': meta['timestamp'],
                            'size_mb': meta.get('size_bytes', 0) / (1024 * 1024)
                        })
                except:
                    pass
            
            return cache_info
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {}
    
    def _get_cache_size(self) -> int:
        """Calculate total cache size in bytes"""
        try:
            total_size = 0
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            return total_size
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries"""
        try:
            removed_count = 0
            expired_keys = []
            
            for key in self.metadata.keys():
                if self._is_expired(key):
                    expired_keys.append(key)
            
            for key in expired_keys:
                if self.remove_cache_entry(key):
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} expired cache entries")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")
            return 0
    
    def remove_cache_entry(self, cache_key: str) -> bool:
        """Remove specific cache entry"""
        try:
            if cache_key not in self.metadata:
                return False
            
            # Remove file
            file_path = self.metadata[cache_key].get('file_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from metadata
            del self.metadata[cache_key]
            self._save_metadata()
            
            logger.info(f"Removed cache entry: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing cache entry: {e}")
            return False
    
    def clear_all_cache(self) -> bool:
        """Clear all cache data"""
        try:
            # Remove all cache files
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                
                # Recreate subdirectories
                for dir_path in [self.data_cache_dir, self.model_cache_dir, 
                               self.results_cache_dir, self.temp_cache_dir]:
                    dir_path.mkdir(exist_ok=True)
            
            # Clear metadata
            self.metadata = {}
            self._save_metadata()
            
            logger.info("Cleared all cache data")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def export_cache_summary(self, output_path: str) -> bool:
        """Export cache summary to file"""
        try:
            cache_info = self.get_cache_info()
            
            summary_lines = [
                "WEATHER PREDICTION PROJECT - CACHE SUMMARY",
                "=" * 50,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Cache Status: {'Enabled' if self.cache_enabled else 'Disabled'}",
                f"Cache Directory: {self.cache_dir}",
                f"Total Entries: {cache_info['total_entries']}",
                f"Total Size: {cache_info['cache_size_mb']:.2f} MB",
                "",
                "Entries by Type:",
            ]
            
            for entry_type, count in cache_info['entries_by_type'].items():
                summary_lines.append(f"  {entry_type}: {count}")
            
            summary_lines.extend([
                "",
                f"Expired Entries: {len(cache_info['expired_entries'])}",
                f"Recent Entries (24h): {len(cache_info['recent_entries'])}",
                "",
                "Recent Cache Activity:"
            ])
            
            for entry in cache_info['recent_entries'][:10]:  # Show last 10
                summary_lines.append(
                    f"  {entry['key']} ({entry['type']}) - {entry['size_mb']:.2f} MB"
                )
            
            summary_text = "\n".join(summary_lines)
            
            with open(output_path, 'w') as f:
                f.write(summary_text)
            
            logger.info(f"Cache summary exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting cache summary: {e}")
            return False

class DatasetCache:
    """Specialized cache for dataset operations"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def cache_city_data(self, city_name: str, data: pd.DataFrame) -> bool:
        """Cache data for a specific city"""
        cache_key = f"city_data_{city_name.lower().replace(' ', '_')}"
        return self.cache_manager.cache_data(data, cache_key)
    
    def load_city_data(self, city_name: str) -> Optional[pd.DataFrame]:
        """Load cached data for a specific city"""
        cache_key = f"city_data_{city_name.lower().replace(' ', '_')}"
        return self.cache_manager.load_cached_data(cache_key)
    
    def cache_processed_features(self, features: pd.DataFrame, processing_params: Dict) -> bool:
        """Cache processed features with processing parameters"""
        params_hash = self.cache_manager._generate_key(**processing_params)
        cache_key = f"processed_features_{params_hash}"
        return self.cache_manager.cache_data(features, cache_key)
    
    def load_processed_features(self, processing_params: Dict) -> Optional[pd.DataFrame]:
        """Load cached processed features"""
        params_hash = self.cache_manager._generate_key(**processing_params)
        cache_key = f"processed_features_{params_hash}"
        return self.cache_manager.load_cached_data(cache_key)

def main():
    """Test cache functionality"""
    try:
        # Initialize cache manager
        cache_manager = CacheManager()
        
        # Test data caching
        test_data = pd.DataFrame({
            'temperature': np.random.normal(25, 5, 100),
            'humidity': np.random.normal(60, 15, 100),
            'city': ['Mumbai'] * 100
        })
        
        # Cache data
        cache_key = "test_data"
        success = cache_manager.cache_data(test_data, cache_key)
        print(f"Data caching success: {success}")
        
        # Load cached data
        loaded_data = cache_manager.load_cached_data(cache_key)
        print(f"Data loading success: {loaded_data is not None}")
        
        if loaded_data is not None:
            print(f"Loaded data shape: {loaded_data.shape}")
        
        # Get cache info
        cache_info = cache_manager.get_cache_info()
        print(f"\nCache Info:")
        for key, value in cache_info.items():
            print(f"  {key}: {value}")
        
        # Export cache summary
        summary_path = "cache/cache_summary.txt"
        cache_manager.export_cache_summary(summary_path)
        print(f"\nCache summary exported to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        raise

if __name__ == "__main__":
    main()
