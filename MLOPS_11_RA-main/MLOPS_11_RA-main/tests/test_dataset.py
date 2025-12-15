import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from steps.ingest import Ingestion

@pytest.fixture
def config_data():
    return {
        'data': {
            'train_path': 'data/train/mushroom_train.csv',
            'test_path': 'data/test/mushroom_test.csv'
        }
    }

@pytest.fixture
def sample_data():
    train_data = pd.DataFrame({
        'class': ['e', 'p'],
        'odor': ['a', 'l'],
        'gill-size': ['b', 'n']
    })
    test_data = pd.DataFrame({
        'class': ['p'],
        'odor': ['p'],
        'gill-size': ['b']
    })
    return train_data, test_data

@patch("builtins.open", new_callable=mock_open, read_data="dummy")
@patch("yaml.safe_load")
@patch("pandas.read_csv")
def test_load_data(mock_read_csv, mock_safe_load, mock_open, config_data, sample_data):
    mock_safe_load.return_value = config_data
    mock_read_csv.side_effect = sample_data

    ingestion = Ingestion()
    train_data, test_data = ingestion.load_data()

    pd.testing.assert_frame_equal(train_data, sample_data[0])
    pd.testing.assert_frame_equal(test_data, sample_data[1])

    mock_read_csv.assert_any_call('data/train/mushroom_train.csv')
    mock_read_csv.assert_any_call('data/test/mushroom_test.csv')
