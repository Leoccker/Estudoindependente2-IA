import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Substitua pelo caminho do seu dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Renomeia colunas se necessário
df = df[['Category', 'Message']]
df.columns = ['Category', 'Message']

print(df.head())
print(df['Category'].value_counts())

# Converter labels (ham=0, spam=1)
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Converter texto para vetores numéricos (Bag of Words)
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['Message']).toarray()

y = df['Category'].values

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]

model = Sequential([
    Dense(32, input_dim=input_dim, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Avaliação no teste
y_pred = (model.predict(X_test) > 0.5).astype("int32")

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.4f}")

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.colorbar()
plt.show()

plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
