# Sistema de predicción y clasificación de oportunidades comerciales

## Metadatos

| Campo | Valor |
|-------|-------|
| **Duración** | 105 minutos |
| **Complejidad** | Alta |
| **Nivel Bloom** | Crear |

## Descripción General

En este laboratorio construirás un sistema completo de predicción y clasificación de oportunidades comerciales utilizando aprendizaje automático supervisado. Implementarás un modelo de regresión lineal para estimar el valor monetario de cada oportunidad y un modelo de regresión logística para clasificar si la oportunidad será ganada o perdida. El flujo integra preprocesamiento de datos, ingeniería de características, entrenamiento con regularización, evaluación con métricas especializadas, visualización interpretativa y persistencia de modelos en disco.

## Objetivos de Aprendizaje

- [ ] Construir un pipeline de preprocesamiento reutilizable que integre imputación, codificación one-hot, escalado y creación de características derivadas
- [ ] Entrenar y evaluar un modelo de regresión lineal (incluyendo Ridge y Lasso) para predecir el valor monetario de oportunidades comerciales
- [ ] Entrenar y evaluar un modelo de regresión logística con análisis de umbral de decisión, curva ROC y frontera de decisión
- [ ] Diagnosticar sobreajuste mediante comparación de métricas train/validation y aplicar regularización como estrategia de mitigación
- [ ] Persistir modelos entrenados y pipeline de preprocesamiento con joblib para simular un flujo de despliegue

## Prerrequisitos

### Conocimiento Requerido

| Concepto | Nivel |
|----------|-------|
| Regresión lineal (hipótesis, MSE, gradiente) | Intermedio |
| Regresión logística (sigmoide, entropía cruzada) | Intermedio |
| Regularización L1/L2 | Básico-Intermedio |
| Python 3.x (funciones, clases, NumPy) | Intermedio |
| Jupyter Notebook (ejecución de celdas, Markdown) | Básico |
| pandas (DataFrames, operaciones vectorizadas) | Básico-Intermedio |

### Acceso Requerido

- Terminal/consola con permisos de escritura en el directorio home del usuario
- Conexión a internet para instalación de paquetes (solo configuración inicial)
- Editor o navegador web para JupyterLab

## Entorno del Laboratorio

### Hardware Mínimo

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| CPU | Intel i5 8ª gen / AMD Ryzen 5 (4 núcleos, 2.0 GHz) | 6+ núcleos |
| RAM | 8 GB | 16 GB |
| Disco libre | 5 GB | 10 GB |
| Pantalla | 1280×768 | 1920×1080 |

### Software Requerido

| Paquete | Versión |
|---------|---------|
| Python | 3.11.9 |
| pip | 24.0 |
| virtualenv | 20.26.2 |
| JupyterLab | 4.2.4 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.1 |
| matplotlib | 3.9.1 |
| seaborn | 0.13.2 |
| scipy | 1.13.1 |
| joblib | 1.4.2 |

### Comandos de Configuración Inicial

```bash
# Crear directorio base del proyecto
mkdir -p ~/ml_lab/lab_01/data
mkdir -p ~/ml_lab/lab_01/models

# Crear y activar entorno virtual
cd ~/ml_lab
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install jupyterlab==4.2.4 numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1 \
    matplotlib==3.9.1 seaborn==0.13.2 scipy==1.13.1 joblib==1.4.2
```

---

## Paso 1: Generación del Dataset y Configuración del Entorno

### Objetivo

Crear el dataset sintético de oportunidades comerciales ejecutando el script generador provisto, verificar la estructura de directorios y lanzar JupyterLab.

### Instrucciones

1. Navega al directorio del laboratorio:

```bash
cd ~/ml_lab/lab_01
```

2. Crea el archivo `generate_dataset.py` con el siguiente contenido (script provisto por el instructor):

```python
"""
generate_dataset.py
Genera un dataset sintético de oportunidades comerciales para el laboratorio.
Ejecutar: python generate_dataset.py
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 2000

# Características categóricas
sectores = ['Tecnología', 'Manufactura', 'Retail', 'Servicios', 'Salud']
canales = ['Directo', 'Partner', 'Inbound', 'Outbound']
regiones = ['Norte', 'Sur', 'Centro', 'Este', 'Oeste']

# Generar características
data = {
    'id_oportunidad': range(1, n_samples + 1),
    'sector_industria': np.random.choice(sectores, n_samples),
    'canal_venta': np.random.choice(canales, n_samples),
    'region': np.random.choice(regiones, n_samples),
    'presupuesto_cliente': np.random.lognormal(mean=10, sigma=0.8, size=n_samples).round(2),
    'tamaño_empresa': np.random.randint(10, 5000, n_samples),
    'num_contactos': np.random.poisson(lam=5, size=n_samples),
    'duracion_ciclo_dias': np.random.gamma(shape=3, scale=20, size=n_samples).round(0).astype(int),
    'score_lead': np.random.uniform(0, 100, n_samples).round(1),
    'num_reuniones': np.random.poisson(lam=3, size=n_samples),
    'antiguedad_relacion_meses': np.random.exponential(scale=12, size=n_samples).round(0).astype(int),
}

df = pd.DataFrame(data)

# Variable objetivo de regresión (valor_oportunidad_usd)
noise = np.random.normal(0, 5000, n_samples)
df['valor_oportunidad_usd'] = (
    0.002 * df['presupuesto_cliente'] +
    8 * df['tamaño_empresa'] +
    1500 * df['num_contactos'] +
    -50 * df['duracion_ciclo_dias'] +
    200 * df['score_lead'] +
    2000 * df['num_reuniones'] +
    100 * df['antiguedad_relacion_meses'] +
    noise
).clip(lower=1000).round(2)

# Variable objetivo de clasificación (oportunidad_ganada)
logit = (
    -2.0 +
    0.03 * df['score_lead'] +
    0.0000005 * df['presupuesto_cliente'] +
    0.15 * df['num_reuniones'] +
    0.1 * df['num_contactos'] +
    -0.005 * df['duracion_ciclo_dias'] +
    0.01 * df['antiguedad_relacion_meses']
)
prob = 1 / (1 + np.exp(-logit))
df['oportunidad_ganada'] = (np.random.uniform(0, 1, n_samples) < prob).astype(int)

# Introducir ~3% de valores nulos en columnas seleccionadas
for col in ['presupuesto_cliente', 'num_contactos', 'duracion_ciclo_dias', 'score_lead']:
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, col] = np.nan

# Guardar dataset
output_path = os.path.join('data', 'commercial_opportunities.csv')
df.to_csv(output_path, index=False)
print(f"Dataset generado exitosamente: {output_path}")
print(f"Registros: {len(df)}, Columnas: {len(df.columns)}")
print(f"Distribución de oportunidad_ganada:\n{df['oportunidad_ganada'].value_counts()}")
```

3. Ejecuta el script generador:

```bash
python generate_dataset.py
```

4. Verifica la estructura de archivos:

```bash
ls -la data/
```

5. Lanza JupyterLab:

```bash
jupyter lab
```

6. En JupyterLab, crea un nuevo notebook llamado `01-00-01_oportunidades_comerciales.ipynb` en el directorio `~/ml_lab/lab_01/`.

### Salida Esperada

```
Dataset generado exitosamente: data/commercial_opportunities.csv
Registros: 2000, Columnas: 13
Distribución de oportunidad_ganada:
oportunidad_ganada
0    1XXX
1     XXX
Name: count, dtype: int64
```

### Verificación

- El archivo `data/commercial_opportunities.csv` existe y tiene un tamaño aproximado de 200-300 KB.
- JupyterLab se abre correctamente en el navegador.
- El notebook `01-00-01_oportunidades_comerciales.ipynb` está creado y vacío.

---

## Paso 2: Carga de Datos y Análisis Exploratorio (FASE 1)

### Objetivo

Cargar el dataset en pandas, inspeccionar su estructura, identificar valores nulos y visualizar distribuciones y correlaciones para comprender los datos antes del modelado.

### Instrucciones

1. En la primera celda del notebook, importa todas las bibliotecas necesarias:

```python
# Celda 1: Importaciones
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
%matplotlib inline

print("Bibliotecas importadas correctamente.")
```

2. Carga el dataset y realiza la inspección inicial:

```python
# Celda 2: Carga y exploración inicial
df = pd.read_csv('data/commercial_opportunities.csv')

print("=" * 60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 60)
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas\n")
df.info()
```

3. Genera estadísticas descriptivas:

```python
# Celda 3: Estadísticas descriptivas
print("\n" + "=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS")
print("=" * 60)
df.describe().round(2)
```

4. Analiza valores nulos:

```python
# Celda 4: Análisis de valores nulos
nulos = df.isnull().sum()
nulos_pct = (df.isnull().sum() / len(df) * 100).round(2)
resumen_nulos = pd.DataFrame({'Nulos': nulos, 'Porcentaje (%)': nulos_pct})
print("\nVALORES NULOS POR COLUMNA:")
print(resumen_nulos[resumen_nulos['Nulos'] > 0])
```

5. Visualiza distribuciones de las variables objetivo:

```python
# Celda 5: Distribuciones de variables objetivo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución de valor_oportunidad_usd
axes[0].hist(df['valor_oportunidad_usd'].dropna(), bins=40, edgecolor='black', alpha=0.7)
axes[0].set_title('Distribución de valor_oportunidad_usd', fontsize=12)
axes[0].set_xlabel('Valor (USD)')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df['valor_oportunidad_usd'].mean(), color='red', linestyle='--', label=f"Media: ${df['valor_oportunidad_usd'].mean():,.0f}")
axes[0].legend()

# Distribución de oportunidad_ganada
counts = df['oportunidad_ganada'].value_counts()
axes[1].bar(['Perdida (0)', 'Ganada (1)'], counts.values, color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[1].set_title('Distribución de oportunidad_ganada', fontsize=12)
axes[1].set_ylabel('Frecuencia')
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 20, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('data/distribucion_objetivos.png', dpi=100, bbox_inches='tight')
plt.show()
```

6. Genera la matriz de correlación:

```python
# Celda 6: Matriz de correlación
cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
cols_numericas.remove('id_oportunidad')

plt.figure(figsize=(12, 9))
corr_matrix = df[cols_numericas].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Matriz de Correlación - Variables Numéricas', fontsize=14)
plt.tight_layout()
plt.show()

print("\nCorrelaciones con valor_oportunidad_usd:")
print(corr_matrix['valor_oportunidad_usd'].sort_values(ascending=False).round(3))
```

### Salida Esperada

- DataFrame con 2000 filas y 13 columnas.
- Aproximadamente 3% de valores nulos en `presupuesto_cliente`, `num_contactos`, `duracion_ciclo_dias` y `score_lead`.
- Histograma de `valor_oportunidad_usd` con distribución aproximadamente normal/lognormal.
- Gráfico de barras mostrando la proporción de oportunidades ganadas vs. perdidas.
- Heatmap de correlaciones con valores numéricos anotados.

### Verificación

- `df.shape` retorna `(2000, 13)`.
- `df.isnull().sum().sum()` retorna un valor entre 150 y 300 (aprox. 3% × 4 columnas × 2000 filas).
- La matriz de correlación muestra que `tamaño_empresa` tiene alta correlación con `valor_oportunidad_usd`.

---

## Paso 3: Preprocesamiento e Ingeniería de Características (FASE 2)

### Objetivo

Implementar un pipeline de preprocesamiento completo que incluya imputación, codificación, escalado y creación de características derivadas, y dividir el dataset en conjuntos de entrenamiento, validación y prueba.

### Instrucciones

1. Define las columnas por tipo y crea características derivadas:

```python
# Celda 7: Definición de columnas e ingeniería de características
# Eliminar columna de ID
df_work = df.drop(columns=['id_oportunidad'])

# Crear características derivadas
df_work['ratio_contactos_duracion'] = (
    df_work['num_contactos'] / df_work['duracion_ciclo_dias'].replace(0, np.nan)
).round(4)

df_work['interaccion_presupuesto_tamaño'] = (
    df_work['presupuesto_cliente'] * df_work['tamaño_empresa']
).round(2)

df_work['reuniones_por_contacto'] = (
    df_work['num_reuniones'] / df_work['num_contactos'].replace(0, np.nan)
).round(4)

print("Características derivadas creadas:")
print("  - ratio_contactos_duracion: num_contactos / duracion_ciclo_dias")
print("  - interaccion_presupuesto_tamaño: presupuesto_cliente × tamaño_empresa")
print("  - reuniones_por_contacto: num_reuniones / num_contactos")
print(f"\nDimensiones actuales: {df_work.shape}")
```

2. Define las listas de columnas:

```python
# Celda 8: Clasificación de columnas
columnas_categoricas = ['sector_industria', 'canal_venta', 'region']
columnas_numericas = [
    'presupuesto_cliente', 'tamaño_empresa', 'num_contactos',
    'duracion_ciclo_dias', 'score_lead', 'num_reuniones',
    'antiguedad_relacion_meses', 'ratio_contactos_duracion',
    'interaccion_presupuesto_tamaño', 'reuniones_por_contacto'
]

# Variables objetivo
target_regresion = 'valor_oportunidad_usd'
target_clasificacion = 'oportunidad_ganada'

print(f"Variables categóricas ({len(columnas_categoricas)}): {columnas_categoricas}")
print(f"Variables numéricas ({len(columnas_numericas)}): {columnas_numericas}")
```

3. Separa las características de las variables objetivo y realiza la división del dataset:

```python
# Celda 9: Separación y división del dataset
X = df_work[columnas_numericas + columnas_categoricas]
y_reg = df_work[target_regresion]
y_clf = df_work[target_clasificacion]

# División: 70% entrenamiento, 15% validación, 15% prueba
X_train_full, X_test, y_reg_train_full, y_reg_test, y_clf_train_full, y_clf_test = \
    train_test_split(X, y_reg, y_clf, test_size=0.15, random_state=42)

X_train, X_val, y_reg_train, y_reg_val, y_clf_train, y_clf_val = \
    train_test_split(X_train_full, y_reg_train_full, y_clf_train_full,
                     test_size=0.176, random_state=42)  # 0.176 × 0.85 ≈ 0.15

print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Conjunto de validación:    {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Conjunto de prueba:        {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
```

4. Construye el pipeline de preprocesamiento:

```python
# Celda 10: Pipeline de preprocesamiento
# Pipeline para variables numéricas: imputación + escalado
pipeline_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para variables categóricas: imputación + codificación one-hot
pipeline_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer que combina ambos pipelines
preprocesador = ColumnTransformer(transformers=[
    ('num', pipeline_numerico, columnas_numericas),
    ('cat', pipeline_categorico, columnas_categoricas)
])

# Ajustar el preprocesador con datos de entrenamiento
X_train_prep = preprocesador.fit_transform(X_train)
X_val_prep = preprocesador.transform(X_val)
X_test_prep = preprocesador.transform(X_test)

# Obtener nombres de columnas transformadas
cat_feature_names = preprocesador.named_transformers_['cat']['onehot'].get_feature_names_out(columnas_categoricas)
feature_names = list(columnas_numericas) + list(cat_feature_names)

print(f"Dimensiones después del preprocesamiento:")
print(f"  X_train_prep: {X_train_prep.shape}")
print(f"  X_val_prep:   {X_val_prep.shape}")
print(f"  X_test_prep:  {X_test_prep.shape}")
print(f"\nTotal de características: {len(feature_names)}")
print(f"  Numéricas: {len(columnas_numericas)}")
print(f"  Categóricas (one-hot): {len(cat_feature_names)}")
```

### Salida Esperada

```
Conjunto de entrenamiento: ~1400 muestras (70.0%)
Conjunto de validación:    ~300 muestras (15.0%)
Conjunto de prueba:        ~300 muestras (15.0%)

Dimensiones después del preprocesamiento:
  X_train_prep: (~1400, 24)
  X_val_prep:   (~300, 24)
  X_test_prep:  (~300, 24)

Total de características: 24
  Numéricas: 10
  Categóricas (one-hot): 14
```

### Verificación

- `X_train_prep.shape[1]` es igual a 24 (10 numéricas + 5 sectores + 4 canales + 5 regiones).
- No hay valores NaN en los arrays preprocesados: `np.isnan(X_train_prep).sum() == 0`.
- Las columnas numéricas tienen media ≈ 0 y desviación estándar ≈ 1 después del escalado.

---

## Paso 4: Modelo de Regresión Lineal para Predicción de Valor (FASE 3)

### Objetivo

Entrenar un modelo de regresión lineal múltiple, implementar manualmente la función de costo MSE, evaluar con métricas estándar, detectar sobreajuste y aplicar regularización Ridge y Lasso.

### Instrucciones

1. Implementa manualmente la función de costo MSE y visualiza conceptualmente la convergencia:

```python
# Celda 11: Implementación manual de función de costo MSE
def calcular_mse_manual(y_real, y_pred):
    """Calcula el Error Cuadrático Medio de forma manual con NumPy."""
    n = len(y_real)
    mse = (1 / n) * np.sum((y_real - y_pred) ** 2)
    return mse

def calcular_rmse_manual(y_real, y_pred):
    """Calcula la Raíz del Error Cuadrático Medio."""
    return np.sqrt(calcular_mse_manual(y_real, y_pred))

# Demostración de descenso del gradiente conceptual (1 característica)
# Usamos la característica más correlacionada para visualización
idx_tamaño = columnas_numericas.index('tamaño_empresa')
X_simple = X_train_prep[:, idx_tamaño].reshape(-1, 1)
y_simple = y_reg_train.values

# Descenso del gradiente manual
np.random.seed(42)
w = np.random.randn(1)
b = 0.0
learning_rate = 0.01
epochs = 100
historial_costo = []

for epoch in range(epochs):
    y_pred = X_simple.flatten() * w[0] + b
    mse = calcular_mse_manual(y_simple, y_pred)
    historial_costo.append(mse)
    
    # Gradientes
    dw = (2 / len(y_simple)) * np.sum((y_pred - y_simple) * X_simple.flatten())
    db = (2 / len(y_simple)) * np.sum(y_pred - y_simple)
    
    # Actualización
    w[0] -= learning_rate * dw
    b -= learning_rate * db

# Visualizar convergencia
plt.figure(figsize=(10, 5))
plt.plot(historial_costo, linewidth=2, color='#2c3e50')
plt.xlabel('Época', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Convergencia del Descenso del Gradiente (1 característica)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"MSE inicial: {historial_costo[0]:,.2f}")
print(f"MSE final:   {historial_costo[-1]:,.2f}")
print(f"Reducción:   {(1 - historial_costo[-1]/historial_costo[0])*100:.1f}%")
```

2. Entrena el modelo de regresión lineal con scikit-learn:

```python
# Celda 12: Entrenamiento de regresión lineal múltiple
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train_prep, y_reg_train)

# Predicciones en los tres conjuntos
y_pred_train_reg = modelo_regresion.predict(X_train_prep)
y_pred_val_reg = modelo_regresion.predict(X_val_prep)
y_pred_test_reg = modelo_regresion.predict(X_test_prep)

# Métricas
metricas_reg = {}
for nombre, y_real, y_pred in [
    ('Entrenamiento', y_reg_train, y_pred_train_reg),
    ('Validación', y_reg_val, y_pred_val_reg),
    ('Prueba', y_reg_test, y_pred_test_reg)
]:
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    metricas_reg[nombre] = {'MSE': mse, 'RMSE': rmse, 'R²': r2}
    print(f"{nombre:15s} | MSE: {mse:>14,.2f} | RMSE: {rmse:>10,.2f} | R²: {r2:.4f}")
```

3. Detecta sobreajuste y aplica regularización:

```python
# Celda 13: Regularización - Ridge y Lasso
print("=" * 70)
print("COMPARACIÓN: LinearRegression vs Ridge vs Lasso")
print("=" * 70)

# Ridge (L2)
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train_prep, y_reg_train)

# Lasso (L1)
modelo_lasso = Lasso(alpha=0.1)
modelo_lasso.fit(X_train_prep, y_reg_train)

# Tabla comparativa
modelos_reg = {
    'LinearRegression': modelo_regresion,
    'Ridge (α=1.0)': modelo_ridge,
    'Lasso (α=0.1)': modelo_lasso
}

resultados = []
for nombre, modelo in modelos_reg.items():
    pred_train = modelo.predict(X_train_prep)
    pred_val = modelo.predict(X_val_prep)
    
    r2_train = r2_score(y_reg_train, pred_train)
    r2_val = r2_score(y_reg_val, pred_val)
    rmse_train = np.sqrt(mean_squared_error(y_reg_train, pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_reg_val, pred_val))
    gap = r2_train - r2_val
    
    resultados.append({
        'Modelo': nombre,
        'R² Train': r2_train,
        'R² Val': r2_val,
        'RMSE Train': rmse_train,
        'RMSE Val': rmse_val,
        'Gap (sobreajuste)': gap
    })

df_resultados_reg = pd.DataFrame(resultados)
print(df_resultados_reg.to_string(index=False))
print(f"\n→ Un gap R² > 0.05 indica posible sobreajuste.")
```

4. Visualiza coeficientes y residuos:

```python
# Celda 14: Visualización de coeficientes y residuos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Coeficientes del modelo Ridge (top 10 por magnitud)
coefs = pd.Series(modelo_ridge.coef_, index=feature_names)
top_coefs = coefs.abs().nlargest(10)
coefs_top = coefs[top_coefs.index].sort_values()

axes[0].barh(range(len(coefs_top)), coefs_top.values, color='steelblue', edgecolor='black')
axes[0].set_yticks(range(len(coefs_top)))
axes[0].set_yticklabels(coefs_top.index, fontsize=9)
axes[0].set_title('Top 10 Coeficientes (Ridge)', fontsize=12)
axes[0].set_xlabel('Valor del coeficiente')

# Residuos vs predicciones (validación)
residuos_val = y_reg_val.values - modelo_ridge.predict(X_val_prep)
axes[1].scatter(modelo_ridge.predict(X_val_prep), residuos_val, alpha=0.5, s=20)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Residuos vs Predicciones (Validación)', fontsize=12)
axes[1].set_xlabel('Valor predicho (USD)')
axes[1].set_ylabel('Residuo')

# Predicho vs Real
axes[2].scatter(y_reg_val, modelo_ridge.predict(X_val_prep), alpha=0.5, s=20)
min_val = min(y_reg_val.min(), modelo_ridge.predict(X_val_prep).min())
max_val = max(y_reg_val.max(), modelo_ridge.predict(X_val_prep).max())
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal')
axes[2].set_title('Predicho vs Real (Validación)', fontsize=12)
axes[2].set_xlabel('Valor real (USD)')
axes[2].set_ylabel('Valor predicho (USD)')
axes[2].legend()

plt.tight_layout()
plt.savefig('data/regresion_diagnostico.png', dpi=100, bbox_inches='tight')
plt.show()
```

### Salida Esperada

- Curva de convergencia del gradiente descendente mostrando reducción progresiva del MSE.
- Tabla comparativa mostrando R² en entrenamiento y validación para los tres modelos.
- Ridge y Lasso deben mostrar un gap menor entre train y validation comparado con LinearRegression (si existe sobreajuste).
- Gráfico de residuos distribuidos aleatoriamente alrededor de cero.
- Gráfico de predicho vs. real con puntos cercanos a la línea diagonal.

### Verificación

- R² en validación es positivo (> 0) para todos los modelos, indicando que los modelos tienen capacidad predictiva.
- Los residuos no muestran patrones sistemáticos (no hay forma de embudo ni curvatura).
- `modelo_ridge.coef_` tiene la misma longitud que `feature_names`.

---

## Paso 5: Modelo de Regresión Logística para Clasificación (FASE 4)

### Objetivo

Entrenar un modelo de regresión logística, implementar manualmente la función sigmoide y la entropía cruzada binaria, evaluar con métricas de clasificación, analizar el umbral de decisión y visualizar la frontera de decisión.

### Instrucciones

1. Implementa manualmente la función sigmoide y la entropía cruzada binaria:

```python
# Celda 15: Implementación manual de sigmoide y entropía cruzada
def sigmoide(z):
    """Función sigmoide: σ(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def entropia_cruzada_binaria(y_real, y_prob):
    """
    Función de costo de entropía cruzada binaria.
    J = -1/m * Σ[y*log(p) + (1-y)*log(1-p)]
    """
    m = len(y_real)
    epsilon = 1e-15  # Evitar log(0)
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    costo = -(1/m) * np.sum(
        y_real * np.log(y_prob) + (1 - y_real) * np.log(1 - y_prob)
    )
    return costo

# Demostración visual de la función sigmoide
z_values = np.linspace(-10, 10, 200)
plt.figure(figsize=(8, 4))
plt.plot(z_values, sigmoide(z_values), linewidth=2.5, color='#8e44ad')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Umbral = 0.5')
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
plt.xlabel('z (combinación lineal)', fontsize=12)
plt.ylabel('σ(z)', fontsize=12)
plt.title('Función Sigmoide', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Verificación numérica
print(f"σ(0) = {sigmoide(0):.4f} (debe ser 0.5)")
print(f"σ(5) = {sigmoide(5):.4f} (debe ser ~0.9933)")
print(f"σ(-5) = {sigmoide(-5):.4f} (debe ser ~0.0067)")
```

2. Entrena el modelo de regresión logística:

```python
# Celda 16: Entrenamiento de regresión logística
modelo_clasificacion = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42,
    C=1.0  # Inverso de la fuerza de regularización
)
modelo_clasificacion.fit(X_train_prep, y_clf_train)

# Predicciones
y_pred_train_clf = modelo_clasificacion.predict(X_train_prep)
y_prob_train_clf = modelo_clasificacion.predict_proba(X_train_prep)[:, 1]
y_pred_val_clf = modelo_clasificacion.predict(X_val_prep)
y_prob_val_clf = modelo_clasificacion.predict_proba(X_val_prep)[:, 1]
y_pred_test_clf = modelo_clasificacion.predict(X_test_prep)
y_prob_test_clf = modelo_clasificacion.predict_proba(X_test_prep)[:, 1]

# Verificar con nuestra función de entropía cruzada
costo_manual = entropia_cruzada_binaria(y_clf_train.values, y_prob_train_clf)
print(f"Entropía cruzada binaria (manual) en entrenamiento: {costo_manual:.4f}")
```

3. Evalúa con métricas de clasificación:

```python
# Celda 17: Evaluación completa del clasificador
print("=" * 70)
print("EVALUACIÓN DEL MODELO DE CLASIFICACIÓN")
print("=" * 70)

for nombre, y_real, y_pred, y_prob in [
    ('Entrenamiento', y_clf_train, y_pred_train_clf, y_prob_train_clf),
    ('Validación', y_clf_val, y_pred_val_clf, y_prob_val_clf),
    ('Prueba', y_clf_test, y_pred_test_clf, y_prob_test_clf)
]:
    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred)
    rec = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    auc = roc_auc_score(y_real, y_prob)
    print(f"\n{nombre}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

# Reporte de clasificación detallado (validación)
print("\n" + "=" * 70)
print("REPORTE DE CLASIFICACIÓN (Validación)")
print("=" * 70)
print(classification_report(y_clf_val, y_pred_val_clf, target_names=['Perdida', 'Ganada']))
```

4. Visualiza la matriz de confusión y la curva ROC:

```python
# Celda 18: Matriz de confusión y curva ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión
cm = confusion_matrix(y_clf_val, y_pred_val_clf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Perdida', 'Ganada'],
            yticklabels=['Perdida', 'Ganada'])
axes[0].set_title('Matriz de Confusión (Validación)', fontsize=12)
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Real')

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_clf_val, y_prob_val_clf)
auc_val = roc_auc_score(y_clf_val, y_prob_val_clf)
axes[1].plot(fpr, tpr, linewidth=2.5, color='#2980b9', label=f'ROC (AUC = {auc_val:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio (AUC = 0.5)')
axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
axes[1].set_title('Curva ROC (Validación)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/clasificacion_evaluacion.png', dpi=100, bbox_inches='tight')
plt.show()
```

5. Analiza el efecto del umbral de decisión:

```python
# Celda 19: Análisis de umbral de decisión
umbrales = np.arange(0.3, 0.71, 0.05)
resultados_umbral = []

for umbral in umbrales:
    y_pred_umbral = (y_prob_val_clf >= umbral).astype(int)
    resultados_umbral.append({
        'Umbral': umbral,
        'Accuracy': accuracy_score(y_clf_val, y_pred_umbral),
        'Precision': precision_score(y_clf_val, y_pred_umbral, zero_division=0),
        'Recall': recall_score(y_clf_val, y_pred_umbral, zero_division=0),
        'F1-Score': f1_score(y_clf_val, y_pred_umbral, zero_division=0)
    })

df_umbrales = pd.DataFrame(resultados_umbral)
print("ANÁLISIS DE UMBRALES DE DECISIÓN:")
print(df_umbrales.to_string(index=False, float_format='%.4f'))

# Visualización
plt.figure(figsize=(10, 5))
plt.plot(df_umbrales['Umbral'], df_umbrales['Precision'], 'o-', label='Precision', linewidth=2)
plt.plot(df_umbrales['Umbral'], df_umbrales['Recall'], 's-', label='Recall', linewidth=2)
plt.plot(df_umbrales['Umbral'], df_umbrales['F1-Score'], '^-', label='F1-Score', linewidth=2)
plt.xlabel('Umbral de decisión', fontsize=12)
plt.ylabel('Métrica', fontsize=12)
plt.title('Trade-off Precision-Recall según Umbral', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Identificar umbral óptimo por F1
mejor_umbral = df_umbrales.loc[df_umbrales['F1-Score'].idxmax(), 'Umbral']
print(f"\n→ Umbral óptimo (máximo F1): {mejor_umbral:.2f}")
```

6. Visualiza la frontera de decisión en 2D:

```python
# Celda 20: Frontera de decisión en espacio 2D
# Seleccionar las 2 características más importantes (mayor coeficiente absoluto)
importancia = np.abs(modelo_clasificacion.coef_[0])
top2_idx = importancia.argsort()[-2:][::-1]
top2_nombres = [feature_names[i] for i in top2_idx]

print(f"Características seleccionadas para visualización 2D: {top2_nombres}")

# Datos 2D
X_2d_train = X_train_prep[:, top2_idx]
X_2d_val = X_val_prep[:, top2_idx]

# Entrenar modelo 2D para visualización
modelo_2d = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
modelo_2d.fit(X_2d_train, y_clf_train)

# Crear malla para frontera de decisión
x_min, x_max = X_2d_val[:, 0].min() - 1, X_2d_val[:, 0].max() + 1
y_min, y_max = X_2d_val[:, 1].min() - 1, X_2d_val[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))
Z = modelo_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Visualizar
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
plt.colorbar(label='P(Ganada)')
scatter = plt.scatter(X_2d_val[:, 0], X_2d_val[:, 1],
                      c=y_clf_val, cmap='RdYlBu', edgecolors='black',
                      linewidth=0.5, s=40, alpha=0.8)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
plt.xlabel(top2_nombres[0], fontsize=12)
plt.ylabel(top2_nombres[1], fontsize=12)
plt.title('Frontera de Decisión (2 características principales)', fontsize=14)
plt.tight_layout()
plt.savefig('data/frontera_decision.png', dpi=100, bbox_inches='tight')
plt.show()
```

7. Visualiza la distribución de probabilidades predichas por clase:

```python
# Celda 21: Distribución de probabilidades predichas
plt.figure(figsize=(10, 5))

# Separar probabilidades por clase real
prob_perdidas = y_prob_val_clf[y_clf_val.values == 0]
prob_ganadas = y_prob_val_clf[y_clf_val.values == 1]

plt.hist(prob_perdidas, bins=30, alpha=0.6, color='#e74c3c', label='Perdidas (real)', density=True)
plt.hist(prob_ganadas, bins=30, alpha=0.6, color='#2ecc71', label='Ganadas (real)', density=True)
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Umbral = 0.5')
plt.xlabel('Probabilidad predicha P(Ganada)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.title('Distribución de Probabilidades Predichas por Clase Real', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Salida Esperada

- Gráfico de la función sigmoide con forma de "S" característica.
- Métricas de clasificación con AUC-ROC > 0.5 (idealmente > 0.65).
- Matriz de confusión mostrando distribución de predicciones correctas e incorrectas.
- Curva ROC por encima de la diagonal.
- Tabla de umbrales mostrando el trade-off precision-recall.
- Frontera de decisión separando las dos clases en el espacio 2D.
- Histogramas de probabilidades predichas con separación visible entre clases.

### Verificación

- `modelo_clasificacion.coef_.shape` es `(1, 24)`.
- AUC-ROC en validación es > 0.5 (el modelo es mejor que el azar).
- La frontera de decisión (línea punteada negra) divide visualmente las regiones de probabilidad en el gráfico 2D.
- Las distribuciones de probabilidades predichas muestran cierta separación entre clases.

---

## Paso 6: Comparación, Persistencia e Interpretación Final (FASE 5)

### Objetivo

Crear una tabla resumen comparativa de ambos modelos, persistir los artefactos en disco con joblib, y verificar la carga y predicción con datos nuevos.

### Instrucciones

1. Genera la tabla resumen comparativa:

```python
# Celda 22: Tabla resumen de ambos modelos
print("=" * 70)
print("RESUMEN COMPARATIVO DE MODELOS")
print("=" * 70)

# Métricas finales en conjunto de prueba
rmse_test = np.sqrt(mean_squared_error(y_reg_test, modelo_ridge.predict(X_test_prep)))
r2_test = r2_score(y_reg_test, modelo_ridge.predict(X_test_prep))

acc_test = accuracy_score(y_clf_test, y_pred_test_clf)
f1_test = f1_score(y_clf_test, y_pred_test_clf)
auc_test = roc_auc_score(y_clf_test, y_prob_test_clf)

resumen = pd.DataFrame({
    'Modelo': ['Ridge Regression (α=1.0)', 'Logistic Regression (C=1.0)'],
    'Tarea': ['Regresión (valor USD)', 'Clasificación (ganada/perdida)'],
    'Métrica Principal': ['RMSE', 'F1-Score'],
    'Valor (Prueba)': [f'${rmse_test:,.2f}', f'{f1_test:.4f}'],
    'Métrica Secundaria': ['R²', 'AUC-ROC'],
    'Valor Secundario': [f'{r2_test:.4f}', f'{auc_test:.4f}']
})

print(resumen.to_string(index=False))
```

2. Persiste los modelos y el pipeline:

```python
# Celda 23: Persistencia de modelos con joblib
import os

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Guardar modelos
joblib.dump(modelo_ridge, os.path.join(models_dir, 'modelo_regresion.joblib'))
joblib.dump(modelo_clasificacion, os.path.join(models_dir, 'modelo_clasificacion.joblib'))
joblib.dump(preprocesador, os.path.join(models_dir, 'pipeline_preprocesamiento.joblib'))

# Verificar archivos guardados
print("Modelos guardados exitosamente:")
for archivo in os.listdir(models_dir):
    filepath = os.path.join(models_dir, archivo)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  {archivo} ({size_kb:.1f} KB)")
```

3. Carga los modelos y realiza predicciones sobre nuevas oportunidades:

```python
# Celda 24: Carga de modelos y predicción sobre datos nuevos
# Simular carga desde disco (como en producción)
preprocesador_cargado = joblib.load('models/pipeline_preprocesamiento.joblib')
modelo_reg_cargado = joblib.load('models/modelo_regresion.joblib')
modelo_clf_cargado = joblib.load('models/modelo_clasificacion.joblib')

# 3 nuevas oportunidades ficticias
nuevas_oportunidades = pd.DataFrame({
    'presupuesto_cliente': [50000, 200000, 15000],
    'tamaño_empresa': [150, 2500, 50],
    'num_contactos': [3, 8, 2],
    'duracion_ciclo_dias': [45, 90, 20],
    'score_lead': [72.5, 88.0, 35.0],
    'num_reuniones': [2, 6, 1],
    'antiguedad_relacion_meses': [6, 24, 1],
    'ratio_contactos_duracion': [3/45, 8/90, 2/20],
    'interaccion_presupuesto_tamaño': [50000*150, 200000*2500, 15000*50],
    'reuniones_por_contacto': [2/3, 6/8, 1/2],
    'sector_industria': ['Tecnología', 'Manufactura', 'Retail'],
    'canal_venta': ['Inbound', 'Directo', 'Outbound'],
    'region': ['Norte', 'Centro', 'Sur']
})

# Preprocesar
X_nuevas = preprocesador_cargado.transform(nuevas_oportunidades)

# Predicciones
valores_predichos = modelo_reg_cargado.predict(X_nuevas)
probabilidades = modelo_clf_cargado.predict_proba(X_nuevas)[:, 1]
clasificaciones = modelo_clf_cargado.predict(X_nuevas)

# Mostrar resultados
print("=" * 70)
print("PREDICCIONES PARA NUEVAS OPORTUNIDADES")
print("=" * 70)
for i in range(3):
    print(f"\nOportunidad {i+1}:")
    print(f"  Sector: {nuevas_oportunidades.iloc[i]['sector_industria']}")
    print(f"  Tamaño empresa: {nuevas_oportunidades.iloc[i]['tamaño_empresa']} empleados")
    print(f"  Score lead: {nuevas_oportunidades.iloc[i]['score_lead']}")
    print(f"  ─────────────────────────────────────")
    print(f"  Valor estimado:        ${valores_predichos[i]:,.2f} USD")
    print(f"  Probabilidad de ganar: {probabilidades[i]*100:.1f}%")
    print(f"  Clasificación:         {'GANADA ✓' if clasificaciones[i] == 1 else 'PERDIDA ✗'}")
    valor_esperado_ajustado = valores_predichos[i] * probabilidades[i]
    print(f"  Valor esperado ajust.: ${valor_esperado_ajustado:,.2f} USD")
```

4. Añade una celda Markdown con conclusiones:

```python
# Celda 25 (Markdown): Conclusiones
```

Contenido Markdown para la celda:

```markdown
## Conclusiones e Interpretación

### Modelo de Regresión (Ridge)
- El modelo Ridge logra un R² razonable en el conjunto de prueba, indicando que las
  características seleccionadas explican una proporción significativa de la variabilidad
  del valor de las oportunidades.
- La regularización L2 redujo el gap entre entrenamiento y validación, mitigando el
  sobreajuste.
- Las características más influyentes son `tamaño_empresa` e `interaccion_presupuesto_tamaño`.

### Modelo de Clasificación (Regresión Logística)
- El AUC-ROC > 0.5 confirma que el modelo discrimina entre oportunidades ganadas y perdidas
  mejor que el azar.
- El análisis de umbrales permite adaptar el modelo según la estrategia comercial:
  umbral bajo → mayor recall (no perder oportunidades), umbral alto → mayor precision
  (enfocar esfuerzo en las más seguras).
- La frontera de decisión muestra una separación lineal entre clases en el espacio de
  características.

### Valor de Negocio
- La combinación de ambos modelos genera un **valor esperado ajustado** que permite
  priorizar la cartera comercial de forma objetiva y reproducible.
- El pipeline de preprocesamiento compartido garantiza consistencia entre ambos modelos.
```

### Salida Esperada

```
Modelos guardados exitosamente:
  modelo_regresion.joblib (X.X KB)
  modelo_clasificacion.joblib (X.X KB)
  pipeline_preprocesamiento.joblib (X.X KB)

PREDICCIONES PARA NUEVAS OPORTUNIDADES
======================================================================

Oportunidad 1:
  Sector: Tecnología
  Tamaño empresa: 150 empleados
  Score lead: 72.5
  ─────────────────────────────────────
  Valor estimado:        $XX,XXX.XX USD
  Probabilidad de ganar: XX.X%
  Clasificación:         GANADA ✓ / PERDIDA ✗
  Valor esperado ajust.: $XX,XXX.XX USD
...
```

### Verificación

- Los tres archivos `.joblib` existen en `~/ml_lab/lab_01/models/`.
- Los modelos cargados producen predicciones numéricas válidas (sin NaN ni errores).
- El valor esperado ajustado es el producto correcto de valor estimado × probabilidad.
- Ejecuta la siguiente comprobación:

```python
# Verificación de integridad
assert os.path.exists('models/modelo_regresion.joblib'), "Modelo de regresión no encontrado"
assert os.path.exists('models/modelo_clasificacion.joblib'), "Modelo de clasificación no encontrado"
assert os.path.exists('models/pipeline_preprocesamiento.joblib'), "Pipeline no encontrado"
assert not np.isnan(valores_predichos).any(), "Predicciones de regresión contienen NaN"
assert all(0 <= p <= 1 for p in probabilidades), "Probabilidades fuera de rango [0,1]"
print("✓ Todas las verificaciones pasaron correctamente.")
```

---

## Validación y Pruebas

Ejecuta las siguientes verificaciones finales para confirmar que el laboratorio se completó correctamente:

```python
# Celda de validación final
print("=" * 70)
print("VALIDACIÓN FINAL DEL LABORATORIO")
print("=" * 70)

checks = []

# 1. Dataset existe y tiene las dimensiones correctas
try:
    df_check = pd.read_csv('data/commercial_opportunities.csv')
    assert df_check.shape == (2000, 13)
    checks.append(("Dataset correcto (2000×13)", True))
except:
    checks.append(("Dataset correcto (2000×13)", False))

# 2. Preprocesador produce dimensiones esperadas
try:
    assert X_train_prep.shape[1] == X_val_prep.shape[1] == X_test_prep.shape[1]
    checks.append(("Dimensiones consistentes entre conjuntos", True))
except:
    checks.append(("Dimensiones consistentes entre conjuntos", False))

# 3. Modelo de regresión tiene R² positivo en test
try:
    r2 = r2_score(y_reg_test, modelo_ridge.predict(X_test_prep))
    assert r2 > 0
    checks.append((f"R² regresión > 0 (R²={r2:.4f})", True))
except:
    checks.append(("R² regresión > 0", False))

# 4. AUC-ROC clasificación > 0.5
try:
    auc = roc_auc_score(y_clf_test, y_prob_test_clf)
    assert auc > 0.5
    checks.append((f"AUC-ROC > 0.5 (AUC={auc:.4f})", True))
except:
    checks.append(("AUC-ROC > 0.5", False))

# 5. Modelos persistidos correctamente
try:
    for f in ['modelo_regresion.joblib', 'modelo_clasificacion.joblib', 'pipeline_preprocesamiento.joblib']:
        assert os.path.exists(os.path.join('models', f))
    checks.append(("3 archivos .joblib en models/", True))
except:
    checks.append(("3 archivos .joblib en models/", False))

# 6. Predicción sobre datos nuevos funciona
try:
    assert len(valores_predichos) == 3
    assert len(probabilidades) == 3
    checks.append(("Predicción sobre 3 nuevas oportunidades OK", True))
except:
    checks.append(("Predicción sobre 3 nuevas oportunidades OK", False))

# Resumen
print()
all_passed = True
for desc, passed in checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {desc}")
    if not passed:
        all_passed = False

print()
if all_passed:
    print("🎉 ¡LABORATORIO COMPLETADO EXITOSAMENTE!")
else:
    print("⚠️  Algunas verificaciones fallaron. Revisa los pasos anteriores.")
```

---

## Solución de Problemas

### Problema 1: Error `ValueError: could not convert string to float` al transformar datos

**Síntomas:** Al ejecutar `preprocesador.transform()` se produce un error indicando que no se puede convertir una cadena a flotante.

**Causa:** Las columnas categóricas se están pasando al pipeline numérico porque el orden de las columnas en el DataFrame no coincide con las listas `columnas_numericas` y `columnas_categoricas` definidas en el `ColumnTransformer`.

**Solución:**

1. Verifica que el DataFrame `X` contenga exactamente las columnas esperadas en el orden correcto:

```python
# Verificar columnas
print("Columnas en X:", X.columns.tolist())
print("Esperadas num:", columnas_numericas)
print("Esperadas cat:", columnas_categoricas)

# Asegurar que X tiene solo las columnas definidas
X = df_work[columnas_numericas + columnas_categoricas]
```

2. Si creaste características derivadas con NaN infinitos (división por cero), reemplázalos antes del preprocesamiento:

```python
df_work.replace([np.inf, -np.inf], np.nan, inplace=True)
```

---

### Problema 2: Error `ConvergenceWarning` en LogisticRegression

**Síntomas:** Al entrenar la regresión logística aparece el warning: `ConvergenceWarning: lbfgs failed to converge (status=1). Increase the number of iterations.`

**Causa:** El optimizador LBFGS no alcanzó la convergencia con el número máximo de iteraciones configurado. Esto suele ocurrir cuando las características no están bien escaladas o cuando hay multicolinealidad alta.

**Solución:**

1. Aumenta `max_iter`:

```python
modelo_clasificacion = LogisticRegression(
    solver='lbfgs',
    max_iter=5000,  # Incrementar de 1000 a 5000
    random_state=42,
    C=1.0
)
```

2. Verifica que el preprocesamiento con `StandardScaler` se aplicó correctamente antes del entrenamiento:

```python
# Verificar que las características están escaladas
print(f"Media de X_train_prep: {X_train_prep.mean(axis=0)[:5]}")  # Debe ser ~0
print(f"Std de X_train_prep:  {X_train_prep.std(axis=0)[:5]}")   # Debe ser ~1
```

3. Si persiste, reduce la fuerza de regularización (aumentar C) o cambiar el solver:

```python
modelo_clasificacion = LogisticRegression(
    solver='saga',  # Alternativa más robusta
    max_iter=5000,
    random_state=42,
    C=0.5,
    penalty='l2'
)
```

---

## Limpieza

Al finalizar el laboratorio, puedes detener JupyterLab y opcionalmente limpiar el entorno:

```bash
# Detener JupyterLab (Ctrl+C en la terminal donde se ejecuta)

# Opcional: desactivar el entorno virtual
deactivate

# Opcional: eliminar artefactos generados (NO recomendado si deseas conservar el trabajo)
# rm -rf ~/ml_lab/lab_01/models/*.joblib
# rm -rf ~/ml_lab/lab_01/data/commercial_opportunities.csv
```

> **Nota:** Se recomienda conservar el notebook completo y los modelos guardados como referencia para futuros laboratorios o proyectos.

---

## Resumen

En este laboratorio has construido un sistema completo de predicción y clasificación de oportunidades comerciales que integra:

| Fase | Logro |
|------|-------|
| Configuración y EDA | Dataset de 2000 registros explorado, distribuciones y correlaciones analizadas |
| Preprocesamiento | Pipeline reutilizable con imputación, codificación one-hot, escalado y 3 características derivadas |
| Regresión | Modelo Ridge entrenado con regularización L2, evaluado con MSE/RMSE/R², residuos visualizados |
| Clasificación | Regresión logística con análisis de umbral, curva ROC, frontera de decisión y distribución de probabilidades |
| Persistencia | 3 artefactos guardados con joblib, carga verificada con predicción sobre datos nuevos |

### Conceptos Clave Reforzados

- La **función de costo MSE** guía el entrenamiento del regresor; la **entropía cruzada binaria** guía al clasificador.
- La **regularización** (Ridge/Lasso) reduce el sobreajuste penalizando coeficientes grandes.
- El **umbral de decisión** permite ajustar el balance precision-recall según la necesidad del negocio.
- El **valor esperado ajustado** (valor × probabilidad) es la métrica de negocio que combina ambos modelos.

### Recursos Adicionales

- [Documentación de scikit-learn: Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Géron, A. — *Hands-On Machine Learning*, Capítulos 4-5 (Regresión y Clasificación)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
- [Documentación de scikit-learn: Métricas de evaluación](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [James, G. et al. — *An Introduction to Statistical Learning*, Capítulo 4](https://www.statlearning.com/)
