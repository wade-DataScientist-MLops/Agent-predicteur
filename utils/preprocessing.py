import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()

    # Supprimer les colonnes avec trop de valeurs manquantes
    df.dropna(thresh=0.5 * len(df), axis=1, inplace=True)

    # Remplir les colonnes numériques avec la médiane
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Remplir les colonnes catégorielles avec le mode
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encoder les colonnes catégorielles
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df
