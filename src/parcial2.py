"""
parcial2.py
Segundo Parcial - Inteligencia Artificial
COTECNOVA - 2026-02
Docente: Jhon James Cano Sánchez
Estudiantes: Brandon Cortes Giraldo & Johan Sttive Linares Barragán
Grupo: 02 (Caña de azúcar - Lotes en Cartago / Valle del Cauca)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética para gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "grupo_02.csv")
DATA_CLEAN = os.path.join(BASE_DIR, "data", "grupo_02_limpio.csv")

def tarea2_carga_e_inspeccion():
    print("=" * 70)
    print("TAREA 2: CARGA E INSPECCIÓN DE DATOS CRUDOS (grupo_02.csv)")
    print("=" * 70)
    
    if not os.path.exists(DATA_RAW):
        # Si se ejecuta dentro de /app en Docker
        alt_raw = os.path.join("/app", "data", "grupo_02.csv")
        if os.path.exists(alt_raw):
            archivo = alt_raw
        else:
            raise FileNotFoundError(f"No se encontró el archivo: {DATA_RAW}")
    else:
        archivo = DATA_RAW
        
    df_raw = pd.read_csv(archivo)
    
    print("\n1. PRIMERAS 5 FILAS DEL DATASET:")
    print(df_raw.head())
    
    print("\n2. FORMA DEL DATAFRAME (shape):")
    print(f"Filas: {df_raw.shape[0]} | Columnas: {df_raw.shape[1]}")
    
    print("\n3. TIPOS DE DATOS (dtypes):")
    print(df_raw.dtypes)
    
    print("\n4. VALORES NULOS POR COLUMNA:")
    print(df_raw.isnull().sum())
    
    print("\n5. FILAS DUPLICADAS:")
    duplicados = df_raw.duplicated().sum()
    print(f"Total filas duplicadas: {duplicados}")
    
    print("\nÚLTIMAS 5 FILAS DEL DATASET (Evidencia de fila de resumen):")
    print(df_raw.tail(5))
    
    return df_raw

def tarea3_calidad_y_limpieza(df_raw):
    print("\n" + "=" * 70)
    print("TAREA 3: CALIDAD Y LIMPIEZA DE DATOS CON PANDAS")
    print("=" * 70)
    
    df = df_raw.copy()
    n_inicial = len(df)
    
    # -------------------------------------------------------------
    # Problema 1: Fila de resumen al final (TOTAL)
    # -------------------------------------------------------------
    fila_total = df[df['lote_id'] == 'TOTAL']
    print(f"\n[Problema 1] Fila de resumen encontrada:\n{fila_total}")
    df = df[df['lote_id'] != 'TOTAL'].copy()
    print("-> Decisión: Fila 'TOTAL' eliminada porque altera las estadísticas reales de los lotes.")
    
    # -------------------------------------------------------------
    # Problema 2: Texto/unidades en columna numérica fertilizante_kg_ha ("371 kg", etc.)
    # -------------------------------------------------------------
    val_texto = df[df['fertilizante_kg_ha'].astype(str).str.contains('kg', case=False, na=False)]
    print(f"\n[Problema 2] Registros con texto en 'fertilizante_kg_ha':\n{val_texto[['lote_id', 'fertilizante_kg_ha']]}")
    
    # Limpieza: eliminar sufijo 'kg' y espacios, convertir a float
    df['fertilizante_kg_ha'] = (
        df['fertilizante_kg_ha']
        .astype(str)
        .str.replace('kg', '', case=False)
        .str.strip()
    )
    df['fertilizante_kg_ha'] = pd.to_numeric(df['fertilizante_kg_ha'], errors='coerce')
    print("-> Decisión: Se eliminó el texto 'kg' y se convirtió la columna a float64.")
    
    # -------------------------------------------------------------
    # Problema 3: Valores nulos en horas_riego_semana
    # -------------------------------------------------------------
    nulos_riego = df[df['horas_riego_semana'].isnull()]
    print(f"\n[Problema 3] Valores nulos en 'horas_riego_semana':\n{nulos_riego[['lote_id', 'vereda', 'horas_riego_semana']]}")
    mediana_riego = df['horas_riego_semana'].median()
    df['horas_riego_semana'] = df['horas_riego_semana'].fillna(mediana_riego)
    print(f"-> Decisión: Imputación con la mediana ({mediana_riego:.1f} horas/semana) para preservar los registros sin sesgar.")
    
    # -------------------------------------------------------------
    # Problema 4: Filas duplicadas
    # -------------------------------------------------------------
    dups = df[df.duplicated(subset=['lote_id'], keep=False)]
    print(f"\n[Problema 4] Filas con lote_id duplicado (total {df.duplicated(subset=['lote_id']).sum()} duplicados):\n{dups[['lote_id', 'vereda', 'horas_riego_semana', 'fertilizante_kg_ha', 'rendimiento_alto']]}")
    df = df.drop_duplicates(subset=['lote_id'], keep='first').copy()
    print("-> Decisión: Se eliminaron las filas duplicadas manteniendo la primera ocurrencia.")
    
    # Asegurar tipo entero en columna objetivo
    df['rendimiento_alto'] = df['rendimiento_alto'].astype(int)
    
    # -------------------------------------------------------------
    # Resumen antes y después
    # -------------------------------------------------------------
    n_final = len(df)
    clase_counts = df['rendimiento_alto'].value_counts()
    
    print("\n--- RESUMEN DE LIMPIEZA ---")
    print(f"Filas antes: {n_inicial} | Filas después: {n_final} (Filas eliminadas: {n_inicial - n_final})")
    print(f"Distribución de clases:")
    print(f"  • Rendimiento Alto (1): {clase_counts.get(1, 0)} lotes ({clase_counts.get(1, 0)/n_final*100:.1f}%)")
    print(f"  • Rendimiento Bajo (0): {clase_counts.get(0, 0)} lotes ({clase_counts.get(0, 0)/n_final*100:.1f}%)")
    
    # Guardar CSV limpio
    out_clean = DATA_CLEAN if os.path.exists(os.path.dirname(DATA_CLEAN)) else "/app/data/grupo_02_limpio.csv"
    df.to_csv(out_clean, index=False)
    print(f"\nDataset limpio exportado a: {out_clean}")
    
    return df

def tarea4_visualizaciones_seaborn(df):
    print("\n" + "=" * 70)
    print("TAREA 4: VISUALIZACIONES ESTADÍSTICAS CON SEABORN")
    print("=" * 70)
    
    # Gráfico 1: Dispersión de las dos variables numéricas frente al objetivo
    plt.figure(figsize=(9, 6))
    palette = {0: "#e11d48", 1: "#10b981"}
    ax1 = sns.scatterplot(
        data=df,
        x="horas_riego_semana",
        y="fertilizante_kg_ha",
        hue="rendimiento_alto",
        style="rendimiento_alto",
        palette=palette,
        s=110,
        alpha=0.9,
        edgecolor="black"
    )
    plt.title("Dispersión: Horas de Riego vs Fertilizante por Rendimiento de Caña", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Horas de Riego por Semana (horas/semana)", fontsize=11, fontweight='bold')
    plt.ylabel("Fertilizante Aplicado (kg/ha)", fontsize=11, fontweight='bold')
    
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend(handles=handles, labels=["Bajo Rendimiento (0)", "Alto Rendimiento (1)"], title="Rendimiento Lote", frameon=True)
    plt.tight_layout()
    plt.savefig("dispersion_riego_fertilizante.png", dpi=200)
    plt.close()
    print("-> Gráfico 1 guardado: dispersion_riego_fertilizante.png")
    
    # Gráfico 2: Boxplot comparando horas de riego según la clase objetivo
    plt.figure(figsize=(8, 5.5))
    sns.boxplot(
        data=df,
        x="rendimiento_alto",
        y="horas_riego_semana",
        palette=["#fda4af", "#a7f3d0"],
        boxprops=dict(alpha=0.85)
    )
    sns.stripplot(
        data=df,
        x="rendimiento_alto",
        y="horas_riego_semana",
        color="black",
        alpha=0.5,
        jitter=0.2,
        size=6
    )
    plt.title("Comparativa de Horas de Riego Semanal según Rendimiento de Caña", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Nivel de Rendimiento (0 = Bajo, 1 = Alto)", fontsize=11, fontweight='bold')
    plt.ylabel("Horas de Riego por Semana", fontsize=11, fontweight='bold')
    plt.xticks([0, 1], ["Rendimiento Bajo (0)", "Rendimiento Alto (1)"])
    plt.tight_layout()
    plt.savefig("boxplot_riego_rendimiento.png", dpi=200)
    plt.close()
    print("-> Gráfico 2 guardado: boxplot_riego_rendimiento.png")

def sigmoide(z):
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))

def loss_logaritmica(y_true, y_pred):
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))

def exactitud(y_true, y_pred):
    clases_pred = (y_pred >= 0.5).astype(int)
    return np.mean(clases_pred == y_true) * 100.0

def tarea5_perceptron_desde_cero(df):
    print("\n" + "=" * 70)
    print("TAREA 5: PERCEPTRÓN DESDE CERO CON NUMPY (SIGMOIDE + GRADIENTE)")
    print("=" * 70)
    
    # 1. Separar X e y
    X = df[['horas_riego_semana', 'fertilizante_kg_ha']].values
    y = df['rendimiento_alto'].values
    m = len(y)
    
    # 2. Estandarización de X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    
    print(f"Parámetros de estandarización:")
    print(f"  • Horas Riego      : Media = {mu[0]:.2f} h/sem | Desv = {sigma[0]:.2f}")
    print(f"  • Fertilizante     : Media = {mu[1]:.2f} kg/ha | Desv = {sigma[1]:.2f}")
    
    # 3. Inicialización
    w = np.zeros(2)
    b = 0.0
    lr = 0.1
    epocas = 500
    
    # Estado inicial
    z_init = X_norm @ w + b
    y_pred_init = sigmoide(z_init)
    loss_init = loss_logaritmica(y, y_pred_init)
    acc_init = exactitud(y, y_pred_init)
    
    print(f"\nEstado Inicial:")
    print(f"  • Pesos iniciales: w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")
    print(f"  • Pérdida inicial: {loss_init:.4f}")
    print(f"  • Exactitud inicial: {acc_init:.2f}%")
    
    hist_loss = []
    hist_acc = []
    
    # 4. Descenso del gradiente
    for epoca in range(epocas):
        z = X_norm @ w + b
        y_pred = sigmoide(z)
        
        # Cálculo de pérdida y exactitud
        loss = loss_logaritmica(y, y_pred)
        acc = exactitud(y, y_pred)
        hist_loss.append(loss)
        hist_acc.append(acc)
        
        # Gradientes
        dz = y_pred - y
        dw = (1.0 / m) * (X_norm.T @ dz)
        db = (1.0 / m) * np.sum(dz)
        
        # Actualización
        w -= lr * dw
        b -= lr * db
    
    # Estado final
    z_final = X_norm @ w + b
    y_pred_final = sigmoide(z_final)
    loss_final = loss_logaritmica(y, y_pred_final)
    acc_final = exactitud(y, y_pred_final)
    
    print(f"\nEstado Final tras {epocas} épocas:")
    print(f"  • Pesos finales: w1 (Riego) = {w[0]:.4f}, w2 (Fertilizante) = {w[1]:.4f}, b = {b:.4f}")
    print(f"  • Pérdida final: {loss_final:.4f} (Reducción de {loss_init - loss_final:.4f})")
    print(f"  • Exactitud final: {acc_final:.2f}%")
    
    # 5. Generar curva_aprendizaje.png
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epocas + 1), hist_loss, color="#2563eb", linewidth=2.5, label="Pérdida Logarítmica (BCE)")
    plt.title("Curva de Aprendizaje del Perceptrón (Caña de Azúcar)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Épocas de Entrenamiento", fontsize=11, fontweight='bold')
    plt.ylabel("Pérdida Logarítmica", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("curva_aprendizaje.png", dpi=200)
    plt.close()
    print("-> Gráfico 3 guardado: curva_aprendizaje.png")
    
    # 6. Generar frontera_decision.png en unidades originales
    # Frontera z = 0 => w1 * ((x1 - mu1)/sigma1) + w2 * ((x2 - mu2)/sigma2) + b = 0
    # Despejando x2:
    # (x2 - mu2)/sigma2 = - (w1 * ((x1 - mu1)/sigma1) + b) / w2
    # x2 = mu2 - sigma2 * (w1 * ((x1 - mu1)/sigma1) + b) / w2
    plt.figure(figsize=(9, 6))
    
    # Puntos de entrenamiento
    mask_1 = (y == 1)
    mask_0 = (y == 0)
    plt.scatter(X[mask_1, 0], X[mask_1, 1], color="#10b981", edgecolors="black", s=110, alpha=0.9, label="Rendimiento Alto (1)")
    plt.scatter(X[mask_0, 0], X[mask_0, 1], color="#e11d48", edgecolors="black", s=110, alpha=0.9, label="Rendimiento Bajo (0)")
    
    x1_vals = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200)
    if abs(w[1]) > 1e-6:
        x2_vals = mu[1] - sigma[1] * (w[0] * ((x1_vals - mu[0]) / sigma[0]) + b) / w[1]
        plt.plot(x1_vals, x2_vals, color="#1e293b", linestyle="--", linewidth=2.5, label="Frontera de Decisión (P = 0.5)")
        
    plt.title("Frontera de Decisión del Perceptrón en Unidades Originales", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Horas de Riego por Semana (h/sem)", fontsize=11, fontweight='bold')
    plt.ylabel("Fertilizante Aplicado (kg/ha)", fontsize=11, fontweight='bold')
    plt.xlim(X[:, 0].min() - 2, X[:, 0].max() + 2)
    plt.ylim(X[:, 1].min() - 15, X[:, 1].max() + 20)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig("frontera_decision.png", dpi=200)
    plt.close()
    print("-> Gráfico 4 guardado: frontera_decision.png")
    
    # Probar 2 casos nuevos requeridos por Tarea 6
    # Caso 1 (esperado 1): Lote con buen riego y fertilización (35 h/sem, 300 kg/ha)
    # Caso 2 (esperado 0): Lote descuidado con poco riego y bajo abono (8 h/sem, 80 kg/ha)
    casos_nuevos = np.array([
        [35.0, 300.0],
        [8.0, 80.0]
    ])
    casos_norm = (casos_nuevos - mu) / sigma
    prob_nuevos = sigmoide(casos_norm @ w + b)
    
    print("\n--- PREDICCIÓN DE CASOS NUEVOS ---")
    print(f"Lote Nuevo 1 (35 h/sem riego, 300 kg/ha fertilizante):")
    print(f"  • Probabilidad calculada: {prob_nuevos[0]*100:.2f}% -> Predicción: {'ALTO RENDIMIENTO (1)' if prob_nuevos[0] >= 0.5 else 'BAJO RENDIMIENTO (0)'}")
    print(f"Lote Nuevo 2 (8 h/sem riego, 80 kg/ha fertilizante):")
    print(f"  • Probabilidad calculada: {prob_nuevos[1]*100:.2f}% -> Predicción: {'ALTO RENDIMIENTO (1)' if prob_nuevos[1] >= 0.5 else 'BAJO RENDIMIENTO (0)'}")
    
    return {
        "mu": mu,
        "sigma": sigma,
        "w": w,
        "b": b,
        "loss_init": loss_init,
        "loss_final": loss_final,
        "acc_init": acc_init,
        "acc_final": acc_final,
        "casos_nuevos": casos_nuevos,
        "prob_nuevos": prob_nuevos
    }

def main():
    print("INICIANDO EJECUCIÓN COMPLETA DEL SEGUNDO PARCIAL...")
    df_raw = tarea2_carga_e_inspeccion()
    df_clean = tarea3_calidad_y_limpieza(df_raw)
    tarea4_visualizaciones_seaborn(df_clean)
    resultados = tarea5_perceptron_desde_cero(df_clean)
    print("\nEJECUCIÓN COMPLETADA EXITOSAMENTE. TODOS LOS ARCHIVOS Y GRÁFICOS HAN SIDO GENERADOS.")

if __name__ == "__main__":
    main()
