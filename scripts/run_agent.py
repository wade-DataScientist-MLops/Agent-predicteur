import os
import pandas as pd
from agent.core import AutoMLAgent
from utils.preprocessing import preprocess
from sklearn.metrics import accuracy_score, classification_report

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(BASE_DIR, "..", "data", "train.csv")
    test_path = os.path.join(BASE_DIR, "..", "data", "test.csv")

    # Chargement et préprocessing des données d'entraînement
    df_train = pd.read_csv(train_path)
    df_train = preprocess(df_train)

    agent = AutoMLAgent(df_train, target_column="Survived")
    agent.run()

    # Chargement et préprocessing des données de test
    df_test = pd.read_csv(test_path)
    df_test = preprocess(df_test)

    if "Survived" in df_test.columns:
        X_test = df_test.drop(columns=["Survived"])
        y_test = df_test["Survived"]
        y_pred = agent.predict(X_test)

        print("[INFO] Évaluation sur données test :")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    else:
        X_test = df_test
        y_pred = agent.predict(X_test)
        print("[INFO] Prédictions sur données test (sans labels) :")
        print(y_pred)

if __name__ == "__main__":
    main()
