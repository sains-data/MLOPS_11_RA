import pytest
import pandas as pd
import numpy as np
from steps.clean import Cleaner

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'class': ['e', 'p', 'e', 'p'],
        'odor': ['a', 'l', '?', 'p'],
        'gill-size': ['b', '?', 'n', 'b'],
        'cap-color': ['n', 'y', 'w', '?']
    })

@pytest.fixture
def cleaner():
    return Cleaner()

def test_clean_data(cleaner, sample_data):
    cleaned_data = cleaner.clean_data(sample_data.copy())

    # Tidak boleh ada missing value
    assert not cleaned_data.isnull().any().any()

    # '?' harus sudah tidak ada
    assert not (cleaned_data == '?').any().any()

    # Jumlah kolom & baris tetap
    assert cleaned_data.shape == sample_data.shape
