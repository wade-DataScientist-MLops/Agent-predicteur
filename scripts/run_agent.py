import pandas as pd
from agent.core import AutoMLAgent

def main():
    df = pd.read_csv("data/train.csv")
    agent = AutoMLAgent(df, target_column="Survived")
    agent.run()

if __name__ == "__main__":
    main()
