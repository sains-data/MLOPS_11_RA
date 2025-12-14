import pytest
import pandas as pd
import numpy as np
from steps.train import Trainer

@pytest.fixture
def sample_mushroom_data():
    """
    Small dummy mushroom dataset
    """
    return pd.DataFrame({
        "class": ["e", "p", "e", "p"],
        "cap-shape": ["x", "x", "b", "f"],
        "odor": ["a", "l", "a", "p"],
        "gill-size": ["b", "n", "b", "n"]
    })

def test_train_pipeline(sample_mushroom_data):
    trainer = Trainer()

    # Split features & target
    X = sample_mushroom_data.drop(columns=["class"])
    y = sample_mushroom_data["class"]

    # Train should run without error
    trainer.pipeline.fit(X, y)

    # Prediction should work
    predictions = trainer.pipeline.predict(X)

    # Assertions
    assert len(predictions) == len(y)
    assert set(predictions).issubset({"e", "p"})
