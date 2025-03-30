import pandas as pd

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    # Map satisfaction
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

    # Drop id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Drop NaN rows (швидкий варіант)
    df = df.dropna()

    return df
