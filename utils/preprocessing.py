import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Supprimer les colonnes avec >50% de valeurs manquantes
    df = df.dropna(thresh=0.5 * len(df), axis=1)

    # Remplir colonnes numériques avec la médiane
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Remplir colonnes catégorielles avec le mode
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encoder colonnes catégorielles
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df
