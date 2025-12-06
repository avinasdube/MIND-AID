import os
import tempfile
import pandas as pd
from src.train import train


def test_phase1_train_smoke():
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "data.csv")
    df = pd.DataFrame({"clean_text": ["a", "b", "c", "d"], "is_depression": [0, 1, 0, 1]})
    df.to_csv(csv, index=False)
    out = os.path.join(tmp, "models")
    train(csv, out)
    assert os.path.exists(os.path.join(out, "tfidf_vectorizer.pkl"))
    assert os.path.exists(os.path.join(out, "svm_model.pkl"))
