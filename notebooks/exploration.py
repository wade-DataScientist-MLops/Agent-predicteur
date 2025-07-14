import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')

# Aperçu du dataset
print(df.head())

# Valeurs manquantes
sns.heatmap(df.isnull(), cbar=False)
plt.title("Valeurs manquantes")
plt.show()

# Répartition des survivants
sns.countplot(x='Survived', data=df)
plt.title("Répartition des survivants")
plt.show()

# Matrice de corrélation
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Corrélation")
plt.show()
