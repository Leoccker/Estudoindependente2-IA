import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, auc
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURAÇÕES GERAIS ---
CONFIG = {
    "data_path": "data/spam.csv",
    "output_dir": "resultados_spam_model",
    "seed": 42,
    "max_features": 5000,
    "test_size": 0.2,
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "threshold": 0.8
}

plt.style.use('ggplot')

def criar_diretorio_saida(caminho):
    """Cria o diretório para salvar os resultados se não existir."""
    if not os.path.exists(caminho):
        os.makedirs(caminho)

def carregar_e_processar_dados(filepath):
    """Carrega o dataset e realiza o pré-processamento básico."""
    try:
        df = pd.read_csv(filepath, encoding="latin-1")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{filepath}' não encontrado.")
        return None, None, None, None, None

    df = df[["Category", "Message"]]
    df.columns = ["Category", "Message"]

    le = LabelEncoder()
    df["Category"] = le.fit_transform(df["Category"])  # ham=0, spam=1

    vectorizer = TfidfVectorizer(
        stop_words="english", 
        lowercase=True, 
        max_features=CONFIG["max_features"]
    )
    X = vectorizer.fit_transform(df["Message"]).toarray()
    y = df["Category"].values

    X_train, X_test, y_train, y_test, txt_train, txt_test = train_test_split(
        X, y, df["Message"], 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["seed"],
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, txt_test

def calcular_pesos_classes(y_train):
    """Calcula pesos para lidar com desbalanceamento de classes."""
    count_neg = np.sum(y_train == 0)
    count_pos = np.sum(y_train == 1)
    total = len(y_train)
    
    weight_0 = (1 / count_neg) * (total / 2.0)
    weight_1 = (1 / count_pos) * (total / 2.0)
    
    return {0: weight_0, 1: weight_1}

def construir_modelo(input_dim):
    """Define a arquitetura da rede neural."""
    model = Sequential([
        Dense(64, input_dim=input_dim, activation="relu", kernel_regularizer="l2"),
        Dropout(0.4),
        Dense(32, activation="relu", kernel_regularizer="l2"),
        Dropout(0.3),
        Dense(16, activation="relu", kernel_regularizer="l2"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        metrics=["accuracy"],
    )
    return model

def plotar_historico(history, output_dir):
    """Plota e salva as curvas de aprendizado."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Treino', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    ax1.set_title('Acurácia do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Treino', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validação', linewidth=2)
    ax2.set_title('Perda (Loss) do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grafico_historico_treinamento.png"))
    plt.close()

def plotar_matriz_confusao(y_test, y_pred_class, output_dir):
    """Plota e salva a matriz de confusão."""
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ham (Normal)', 'Spam'],
                yticklabels=['Ham (Normal)', 'Spam'],
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Matriz de Confusão', fontsize=14, pad=20)
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grafico_matriz_confusao.png"))
    plt.close()

def plotar_curvas_metricas(y_test, y_pred_proba, output_dir):
    """Plota Curva ROC e Curva Precision-Recall."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Curva ROC')
    ax1.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Curva Precision-Recall')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grafico_curvas_roc_pr.png"))
    plt.close()

def gerar_relatorio_texto(y_test, y_pred_class, y_pred_proba, history, output_dir):
    """Gera um arquivo de texto com as métricas essenciais."""
    acc = accuracy_score(y_test, y_pred_class)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred_class)
    
    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    filepath = os.path.join(output_dir, "relatorio.txt")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("SPAM DETECTOR\n\n")
        
        f.write("RESUMO\n")
        f.write(f"Acurácia:   {acc:.4f}\n")
        f.write(f"ROC-AUC:    {roc_auc:.4f}\n")
        f.write(f"MCC:        {mcc:.4f}\n")
        f.write(f"FPR:        {fpr:.2%}\n")
        f.write(f"Precisão:   {precision:.2%}\n\n")
        
        f.write("MATRIZ DE CONFUSÃO\n")
        f.write(f"TN: {tn:<5} FP: {fp}\n")
        f.write(f"FN: {fn:<5} TP: {tp}\n\n\n")
        
        f.write(classification_report(y_test, y_pred_class, target_names=["Ham", "Spam"]))
        f.write("\n")
        
        f.write("HISTÓRICO (Últimas 5 épocas)\n")
        f.write(f"{'Época':<8} {'Loss T':<10} {'Loss V':<10} {'Acc V':<10}\n")
        
        hist_acc = history.history['accuracy']
        hist_val_acc = history.history['val_accuracy']
        hist_loss = history.history['loss']
        hist_val_loss = history.history['val_loss']
        
        start_idx = max(0, len(hist_acc) - 5)
        for i in range(start_idx, len(hist_acc)):
            f.write(f"{i+1:<8} {hist_loss[i]:<10.4f} {hist_val_loss[i]:<10.4f} {hist_val_acc[i]:<10.4f}\n")

    print(f"Relatório salvo em: {filepath}")

def salvar_erros_csv(txt_test, y_test, y_pred_class, output_dir):
    """Salva um CSV contendo apenas as mensagens onde o modelo errou."""
    df_test = pd.DataFrame({
        "Mensagem": txt_test,
        "Real": y_test,
        "Predito": y_pred_class
    })

    label_map = {0: "Ham", 1: "Spam"}
    df_test["Real_Label"] = df_test["Real"].map(label_map)
    df_test["Predito_Label"] = df_test["Predito"].map(label_map)

    erros = df_test[df_test["Real"] != df_test["Predito"]]

    erros["Mensagem"] = erros["Mensagem"].astype(str).str.replace(r'[\n\r]+', ' ', regex=True)
    
    filepath = os.path.join(output_dir, "analise_erros.csv")
    erros[["Real_Label", "Predito_Label", "Mensagem"]].to_csv(filepath, index=False)

def main():
    criar_diretorio_saida(CONFIG["output_dir"])

    X_train, X_test, y_train, y_test, txt_test = carregar_e_processar_dados(CONFIG["data_path"])
    
    if X_train is None:
        return

    class_weights = calcular_pesos_classes(y_train)

    model = construir_modelo(input_dim=X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, 
        epochs=CONFIG["epochs"], 
        batch_size=CONFIG["batch_size"],
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_proba = model.predict(X_test).flatten()
    y_pred_class = (y_pred_proba > CONFIG["threshold"]).astype("int32")

    plotar_historico(history, CONFIG["output_dir"])
    plotar_matriz_confusao(y_test, y_pred_class, CONFIG["output_dir"])
    plotar_curvas_metricas(y_test, y_pred_proba, CONFIG["output_dir"])
    
    gerar_relatorio_texto(y_test, y_pred_class, y_pred_proba, history, CONFIG["output_dir"])
    
    salvar_erros_csv(txt_test, y_test, y_pred_class, CONFIG["output_dir"])
    
    print("\n" + "="*50)
    print(f"PROCESSO CONCLUÍDO COM SUCESSO.")
    print("="*50)

if __name__ == "__main__":
    main()
