import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Carregue o dataset (pode mudar para o white se quiser)
df = pd.read_csv("winequality-red.csv", sep=";")

# Separar variáveis preditoras e alvo
X = df.drop("quality", axis=1)
y = df["quality"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Criar e treinar o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliar o modelo
y_pred = modelo.predict(X_test)
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))

# Salvar o modelo
joblib.dump(modelo, "modelo_vinho_red.pkl")
print("Modelo salvo como modelo_vinho_red.pkl")
