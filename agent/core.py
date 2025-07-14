import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from utils.preprocessing import preprocess_dataframe

class AutoMLAgent:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        self.model = None

    def train_model(self):
        print("[INFO] Preprocessing...")
        df_clean = preprocess_dataframe(self.df, self.target_column)

        print("[INFO] Splitting dataset...")
        X = df_clean.drop(columns=[self.target_column])
        y = df_clean[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("[INFO] Training RandomForestClassifier...")
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print("[INFO] Evaluation:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def run(self):
        self.train_model()
