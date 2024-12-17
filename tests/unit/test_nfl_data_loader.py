"""Unit tests for NFLDataLoader class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.nfl_data_loader import NFLDataLoader


class TestNFLDataLoader:
    """Test suite for NFLDataLoader."""
    
    @pytest.fixture
    def loader(self, tmp_path):
        """Create a loader instance with temporary cache directory."""
        return NFLDataLoader(cache_dir=tmp_path / "cache")
    
    @pytest.fixture
    def sample_pbp_data(self):
        """Create sample play-by-play data."""
        return pd.DataFrame({
            'game_id': ['2023_01_ARI_BUF', '2023_01_ARI_BUF'],
            'play_id': [1, 2],
            'yards_gained': [5, -2],
            'play_type': ['run', 'pass']
        })
    
    def test_initialization(self, tmp_path):
        """Test loader initialization."""
        cache_dir = tmp_path / "test_cache"
        loader = NFLDataLoader(cache_dir=cache_dir)
        
        assert loader.cache_dir == cache_dir
        assert cache_dir.exists()
    
    def test_default_cache_dir(self):
        """Test default cache directory creation."""
        loader = NFLDataLoader()
        assert loader.cache_dir == Path("data/raw")
    
    @patch('data.nfl_data_loader.nfl')
    def test_load_pbp_data_success(self, mock_nfl, loader, sample_pbp_data):
        """Test successful play-by-play data loading."""
        mock_nfl.import_pbp_data.return_value = sample_pbp_data
        
        result = loader.load_pbp_data([2023])
        
        mock_nfl.import_pbp_data.assert_called_once_with([2023], columns=None)
        assert len(result) == 2
        assert 'play_type' in result.columns
        
        # Check if data was cached
        cache_file = loader.cache_dir / "pbp_2023.parquet"
        assert cache_file.exists()
    
    @patch('data.nfl_data_loader.nfl')
    def test_load_pbp_data_with_columns(self, mock_nfl, loader, sample_pbp_data):
        """Test loading specific columns."""
        mock_nfl.import_pbp_data.return_value = sample_pbp_data[['game_id', 'play_id']]
        
        columns = ['game_id', 'play_id']
        result = loader.load_pbp_data([2023], columns=columns)
        
        mock_nfl.import_pbp_data.assert_called_once_with([2023], columns=columns)
        assert list(result.columns) == columns
    
    @patch('data.nfl_data_loader.nfl')
    def test_load_schedules(self, mock_nfl, loader):
        """Test schedule loading."""
        sample_schedule = pd.DataFrame({
            'game_id': ['2023_01_ARI_BUF'],
            'week': [1],
            'home_team': ['BUF'],
            'away_team': ['ARI']
        })
        mock_nfl.import_schedules.return_value = sample_schedule
        
        result = loader.load_schedules([2023])
        
        mock_nfl.import_schedules.assert_called_once_with([2023])
        assert len(result) == 1
        assert result.iloc[0]['home_team'] == 'BUF'
    
    @patch('data.nfl_data_loader.nfl')
    def test_load_team_stats(self, mock_nfl, loader):
        """Test team statistics loading."""
        sample_stats = pd.DataFrame({
            'player_id': ['00-0036971'],
            'player_name': ['Josh Allen'],
            'recent_team': ['BUF'],
            'season': [2023]
        })
        mock_nfl.import_weekly_data.return_value = sample_stats
        
        result = loader.load_team_stats([2023])
        
        mock_nfl.import_weekly_data.assert_called_once_with([2023])
        assert len(result) == 1
    
    @patch('data.nfl_data_loader.nfl')
    def test_load_rosters(self, mock_nfl, loader):
        """Test roster loading."""
        sample_roster = pd.DataFrame({
            'player_id': ['00-0036971'],
            'player_name': ['Josh Allen'],
            'team': ['BUF'],
            'position': ['QB']
        })
        mock_nfl.import_rosters.return_value = sample_roster
        
        result = loader.load_rosters([2023])
        
        mock_nfl.import_rosters.assert_called_once_with([2023])
        assert len(result) == 1
        assert result.iloc[0]['position'] == 'QB'
    
    def test_get_cached_data_exists(self, loader, sample_pbp_data):
        """Test retrieving existing cached data."""
        # Create cache file
        cache_file = loader.cache_dir / "pbp_2023.parquet"
        sample_pbp_data.to_parquet(cache_file)
        
        result = loader.get_cached_data("pbp", [2023])
        
        assert result is not None
        assert len(result) == 2
    
    def test_get_cached_data_not_exists(self, loader):
        """Test retrieving non-existent cached data."""
        result = loader.get_cached_data("pbp", [2023])
        assert result is None
    
    @patch('data.nfl_data_loader.nfl')
    def test_import_error_handling(self, mock_nfl, loader):
        """Test handling of import errors."""
        mock_nfl.import_pbp_data.side_effect = ImportError("nfl_data_py not installed")
        
        with pytest.raises(ImportError):
            loader.load_pbp_data([2023])
    
    @patch('data.nfl_data_loader.nfl')
    def test_general_error_handling(self, mock_nfl, loader):
        """Test handling of general errors."""
        mock_nfl.import_pbp_data.side_effect = Exception("Network error")
        
        with pytest.raises(Exception) as exc_info:
            loader.load_pbp_data([2023])
        
        assert "Network error" in str(exc_info.value)
