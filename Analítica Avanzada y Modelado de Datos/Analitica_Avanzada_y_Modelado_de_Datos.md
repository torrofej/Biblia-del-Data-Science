# **Analítica Avanzada y Modelado de Datos**

---

## **Índice**

- [1. Extracción del Conocimiento: el Proceso KDD](#1-extracción-del-conocimiento-el-proceso-kdd)
  - [Fases del proceso KDD](#fases-del-proceso-kdd)
- [2. Tipos de Problemas de Machine Learning](#2-tipos-de-problemas-de-machine-learning)
  - [2.1. Aprendizaje Supervisado](#21-aprendizaje-supervisado)
  - [2.2. Aprendizaje No Supervisado](#22-aprendizaje-no-supervisado)
  - [2.3. Aprendizaje Semisupervisado](#23-aprendizaje-semisupervisado)
- [3. Inferencia vs Predicción](#3-inferencia-vs-predicción)
- [4. Scikit-learn y el Ecosistema Python para ML](#4-scikit-learn-y-el-ecosistema-python-para-ml)
- [5. Entrenamiento y Validación de Modelos](#5-entrenamiento-y-validación-de-modelos)
  - [Enfoques de partición](#enfoques-de-partición)
  - [Datos no representativos: desbalanceo de clases](#datos-no-representativos-desbalanceo-de-clases)
- [6. Overfitting, Underfitting, Bias y Varianza](#6-overfitting-underfitting-bias-y-varianza)
  - [6.1. Overfitting (Sobreajuste)](#61-overfitting-sobreajuste)
  - [6.2. Underfitting (Infraajuste)](#62-underfitting-infraajuste)
  - [6.3. Bias (Sesgo) — Error Sistemático](#63-bias-sesgo--error-sistemático)
  - [6.4. Variance (Varianza) — Sensibilidad al Ruido](#64-variance-varianza--sensibilidad-al-ruido)
  - [6.5. Resumen: el trade-off Bias-Varianza](#65-resumen-el-trade-off-bias-varianza)
- [7. LOOCV y Ajuste de Hiperparámetros](#7-loocv-y-ajuste-de-hiperparámetros)
  - [7.1. LOOCV (Leave-One-Out Cross Validation)](#71-loocv-leave-one-out-cross-validation)
  - [7.2. Grid Search: Ajuste de Hiperparámetros](#72-grid-search-ajuste-de-hiperparámetros)
- [8. Métricas de Evaluación](#8-métricas-de-evaluación)
  - [8.1. Distinción clave: Validación vs Evaluación](#81-distinción-clave-validación-vs-evaluación)
  - [8.2. Métricas para Clasificación](#82-métricas-para-clasificación)
    - [Matriz de Confusión](#matriz-de-confusión)
    - [Métricas derivadas](#métricas-derivadas)
    - [Curva ROC y AUC](#curva-roc-y-auc)
  - [8.3. Métricas para Regresión](#83-métricas-para-regresión)
  - [8.4. Métricas para Clustering (No Supervisado)](#84-métricas-para-clustering-no-supervisado)
    - [Método del Codo (Elbow Method)](#método-del-codo-elbow-method)
    - [Coeficiente de Silueta](#coeficiente-de-silueta)

---

## **1. Extracción del Conocimiento: el Proceso KDD**

El **KDD** (*Knowledge Discovery in Databases*) es el proceso completo de extracción de conocimiento útil, válido y comprensible a partir de datos. No es simplemente aplicar un modelo: es un flujo metodológico end-to-end.

### **Fases del proceso KDD**

| Fase | Qué se hace |
|---|---|
| **1. Selección** | Elegir variables, períodos, segmentos o áreas geográficas relevantes para el problema |
| **2. Preprocesamiento** | Limpieza de datos, tratamiento de nulos, eliminación de ruido e inconsistencias |
| **3. Transformación** | Feature engineering, normalización/estandarización, reducción de dimensionalidad |
| **4. Modelización** | Aplicación de algoritmos: regresión, clasificación, clustering, reglas de asociación |
| **5. Interpretación y evaluación** | Analizar resultados, validar que tienen sentido de negocio y traducirlos en conocimiento útil |

> ⚠️ El proceso es **iterativo**, no lineal. Los resultados de la fase 5 frecuentemente obligan a volver a fases anteriores para ajustar la selección de variables o el preprocesamiento.

---

## **2. Tipos de Problemas de Machine Learning**

Los problemas de ML se clasifican según si se dispone o no de etiquetas en los datos de entrenamiento.

### **2.1. Aprendizaje Supervisado**

El algoritmo aprende a partir de un conjunto de datos **etiquetado**: se conoce `y` para cada observación y el objetivo es aprender la relación `X → y` para predecir sobre nuevos datos.

**Regresión** — variable objetivo numérica continua (precio, temperatura, volumen de ventas).

**Clasificación** — variable objetivo categórica (¿comprará?, ¿es fraude?, ¿es spam?).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Clasificación
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Regresión
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

> 📌 **Patrón universal de scikit-learn**: `fit(X_train, y_train)` para entrenar, `predict(X_test)` para inferir. Todos los modelos comparten esta interfaz.

### **2.2. Aprendizaje No Supervisado**

No existe variable objetivo. El objetivo es **descubrir la estructura subyacente** de los datos: agrupaciones, patrones ocultos o simplificaciones no visibles a simple vista.

**Clustering** — agrupa observaciones similares entre sí y distintas del resto. Algoritmo más usado: K-Means.

**Reglas de asociación** — detecta relaciones "si ocurre A, suele ocurrir B". Muy usadas en *market basket analysis*. Algoritmo: A priori.

**Reducción de dimensionalidad** — sintetiza muchas variables en pocas sin perder información crítica. Útil con regresores altamente correlados. Algoritmo más usado: PCA.

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Reducción de dimensionalidad
pca = PCA(n_components=2)
X_reducido = pca.fit_transform(X)
```

### **2.3. Aprendizaje Semisupervisado**

Combina supervisado y no supervisado cuando se dispone de **pocos datos etiquetados** y muchos sin etiquetar. Etiquetar manualmente toda la población es difícil, costoso o imposible.

**Self-Training** — proceso iterativo en tres fases:
1. Entrenar un modelo base con los datos etiquetados disponibles.
2. Predecir sobre los no etiquetados.
3. Incorporar al entrenamiento las predicciones con mayor probabilidad (p. ej. > 95%) y re-entrenar. Repetir 3-4 iteraciones.

**Cluster-then-label** — clusterizar toda la población y asignar la etiqueta mayoritaria del cluster a los miembros no etiquetados.

**Label Propagation** — conecta los datos por una red de similitud; las etiquetas conocidas se propagan hacia los nodos vecinos no etiquetados.

```python
from sklearn.semi_supervised import LabelPropagation
import numpy as np

# -1 indica observación NO etiquetada
y_semi = np.copy(y_train)
y_semi[indices_no_etiquetados] = -1

lp = LabelPropagation()
lp.fit(X_train, y_semi)
y_pred = lp.predict(X_test)
```

---

## **3. Inferencia vs Predicción**

Dos objetivos distintos que determinan qué modelo usar y cómo interpretarlo.

| | Inferencia | Predicción |
|---|---|---|
| **Pregunta** | ¿Cómo y cuánto influye cada variable en el resultado? | ¿Cuál será el valor con el menor error posible? |
| **Prioridad** | Interpretabilidad y causalidad | Exactitud |
| **Modelos** | Regresión lineal/logística (coeficientes interpretables) | Boosting, redes neuronales (caja negra) |
| **Casos de uso** | Análisis de sensibilidad, políticas de negocio | Logística, detección de fraude, bolsa |

**Inferencia** — el modelo cuantifica el efecto de cada variable. Un coeficiente β negativo en una regresión de ventas indica que subir el precio las reduce, y cuantifica ese impacto.

```python
import statsmodels.api as sm

X_con_cte = sm.add_constant(X_train)
modelo = sm.OLS(y_train, X_con_cte).fit()
print(modelo.summary())  # coeficientes, p-valores, intervalos de confianza
```

**Predicción** — el modelo maximiza la exactitud sin necesidad de ser interpretable.

```python
from sklearn.ensemble import GradientBoostingRegressor

modelo = GradientBoostingRegressor(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

---

## **4. Scikit-learn y el Ecosistema Python para ML**

**Scikit-learn** es la librería de referencia para ML tradicional en Python. Open source, construida sobre NumPy, SciPy y Matplotlib. Cubre el ciclo completo: preprocesamiento, modelado, evaluación y selección de modelos.

### **Cuándo usar cada librería**

| Librería | Cuándo usarla |
|---|---|
| **Scikit-learn** | ML tradicional: clasificación, regresión, clustering, reducción de dimensionalidad |
| **XGBoost / LightGBM / CatBoost** | Máxima precisión en datos tabulares; competiciones o entornos de alta exigencia |
| **PyTorch / TensorFlow-Keras** | Deep Learning, datos no estructurados (imágenes, texto, audio), arquitecturas avanzadas |
| **Statsmodels** | Inferencia estadística pura, contraste de hipótesis, análisis econométrico |
| **Prophet** | Forecasting de series temporales con estacionalidad y efectos de calendario |

---

## **5. Entrenamiento y Validación de Modelos**

En ciencia de datos se subdivide el conjunto de datos para garantizar que el modelo **generaliza** correctamente, no solo memoriza los datos de entrenamiento.

### **Enfoques de partición**

**Clásico — Train/Test (80/20)**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Riguroso — Train/Validation/Test (70/15/15)**

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

**Robusto — Validación Cruzada (K-Fold)**

Una única partición 80/20 puede estar sesgada por azar. K-Fold repite el proceso K veces rotando el bloque de test. El rendimiento final es el promedio de las K iteraciones.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"RMSE medio: {(-scores.mean())**0.5:.4f}")
```

### **Datos no representativos: desbalanceo de clases**

Si la variable objetivo está desequilibrada, un muestreo aleatorio simple producirá un modelo sesgado. La solución es **balancear** antes de entrenar.

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_train, y_train)
```

| Técnica | Estrategia |
|---|---|
| **SMOTE** | Genera muestras sintéticas interpolando entre casos minoritarios |
| **ADASYN** | Igual que SMOTE pero focaliza la generación en zonas difíciles de clasificar |
| **Random Oversampling** | Duplica muestras minoritarias aleatoriamente |

---

## **6. Overfitting, Underfitting, Bias y Varianza**

### **6.1. Overfitting (Sobreajuste)**

> El modelo aprende los datos de entrenamiento **demasiado bien**, incluyendo su ruido.

| | |
|---|---|
| **Causa** | Modelo excesivamente complejo para el volumen de datos disponible |
| **Síntoma** | Error muy bajo en train (~10%) y alto en test (~40%) |
| **Indicador clave** | Brecha grande entre métricas de train y test |
| **Solución** | Regularización, más datos, reducir complejidad del modelo |

### **6.2. Underfitting (Infraajuste)**

> El modelo es **demasiado simple** para capturar la estructura de los datos.

| | |
|---|---|
| **Causa** | Modelo sub-parametrizado, variables relevantes omitidas |
| **Síntoma** | Error elevado tanto en train como en test |
| **Indicador clave** | Ambas métricas son malas por igual |
| **Solución** | Aumentar complejidad del modelo, añadir variables relevantes |

### **6.3. Bias (Sesgo) — Error Sistemático**

> El modelo falla **de la misma forma** independientemente de los datos.

| | |
|---|---|
| **Causa** | Suposición errónea sobre la distribución, modelo demasiado rígido |
| **Síntoma** | Distancia sistemática entre predicción media y valor real |
| **Ejemplo** | Aproximar una curva de puntos con una línea recta |

### **6.4. Variance (Varianza) — Sensibilidad al Ruido**

> El modelo cambia sus estimaciones drásticamente ante pequeñas variaciones en los datos de entrada.

| | |
|---|---|
| **Causa** | Sobre-parametrización, datasets pequeños con modelos muy complejos, outliers |
| **Síntoma** | Excelente en train, pésimo en test |
| **Ejemplo** | Polinomio de grado 20 o red neuronal profunda sin regularización |

### **6.5. Resumen: el trade-off Bias-Varianza**

| | **Bias alto** | **Bias bajo** |
|---|---|---|
| **Varianza alta** | Peor escenario posible | Overfitting |
| **Varianza baja** | Underfitting | Modelo ideal ✓ |

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    modelo, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
# Si train >> val  → overfitting (alta varianza)
# Si ambos bajos   → underfitting (alto sesgo)
```

---

## **7. LOOCV y Ajuste de Hiperparámetros**

### **7.1. LOOCV (Leave-One-Out Cross Validation)**

Caso extremo de K-Fold donde K = N (número de filas del dataset). Cada iteración entrena con N-1 filas y valida con 1. Útil cuando el dataset es muy pequeño.

| | |
|---|---|
| **Cuándo usarlo** | Datasets muy pequeños (bioestadística, lanzamiento de nuevos productos) |
| **Ventaja** | Aprovecha casi todas las muestras para entrenar |
| **Desventaja** | Computacionalmente muy costoso; alta varianza si el modelo es sensible |

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
scores = cross_val_score(modelo, X, y, cv=loo, scoring='neg_mean_absolute_error')
print(f"MAE medio: {-scores.mean():.4f}")
```

### **7.2. Grid Search: Ajuste de Hiperparámetros**

Técnica de fuerza bruta sistemática para encontrar la combinación óptima de hiperparámetros según una métrica objetivo. Se combina con K-Fold para que los resultados sean fiables.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeRegressor(),
    param_grid,
    cv=5,                          # K-Fold interno
    scoring='neg_mean_squared_error'
)
grid.fit(X_train, y_train)

print(grid.best_params_)           # {'max_depth': 6, 'min_samples_split': 5}
print(grid.best_score_)
```

> 📌 Si defines 2 hiperparámetros con 5 valores cada uno → 25 modelos × K folds = 125 entrenamientos. Con modelos complejos, usar `RandomizedSearchCV` es más eficiente.

---

## **8. Métricas de Evaluación**

### **8.1. Distinción clave: Validación vs Evaluación**

- **Validación**: medir el comportamiento del modelo durante el entrenamiento para ajustar hiperparámetros y seleccionar el mejor modelo.
- **Evaluación**: medición final del rendimiento sobre el **test set**, que nunca ha sido visto durante el entrenamiento ni la validación.

---

### **8.2. Métricas para Clasificación**

### **Matriz de Confusión**

Tabla que compara predicciones vs realidad, desglosando aciertos y tipos de error.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

### - Métricas derivadas

| Métrica | Fórmula | Cuándo priorizarla |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Clases balanceadas |
| **Precision** | TP / (TP + FP) | Minimizar falsos positivos (ej. spam) |
| **Recall** | TP / (TP + FN) | Minimizar falsos negativos (ej. diagnóstico médico) |
| **F1-Score** | 2 · (P · R) / (P + R) | Equilibrio entre Precision y Recall |

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

### - Curva ROC y AUC

La curva ROC representa el rendimiento del modelo en **todos los umbrales de decisión posibles**, comparando la tasa de Verdaderos Positivos vs la tasa de Falsos Positivos. El **AUC** (área bajo la curva) resume esto en un único número entre 0 y 1.

> ⚠️ El umbral óptimo no es siempre 0.5 — depende del coste de cada tipo de error en el contexto de negocio.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
auc = roc_auc_score(y_test, y_prob[:, 1])

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1],[0,1],'--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.show()
```

---

### **8.3. Métricas para Regresión**

Todas se basan en medir la distancia entre el valor real y el valor predicho.

| Métrica | Fórmula conceptual | Característica |
|---|---|---|
| **MAE** | mean(\|y - ŷ\|) | Fácil de interpretar, robusto a outliers |
| **MSE** | mean((y - ŷ)²) | Penaliza errores grandes |
| **RMSE** | √MSE | Error en la unidad original de la variable |
| **MAPE** | mean(\|y - ŷ\| / y) · 100 | Interpretable como porcentaje |
| **R²** | 1 - SCE/SCT | Proporción de varianza explicada por el modelo (0–1) |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f} | MAPE: {mape:.1f}%")
```

> ⚠️ **MAPE explota si hay valores reales igual a 0** (división por cero). En ese caso usar MAE o RMSE.

---

### **8.4. Métricas para Clustering (No Supervisado)**

### - Método del Codo (Elbow Method)

Para cada valor de K, calcula el **WCSS** (Within-Cluster Sum of Squares — suma de distancias de cada punto a su centroide). Se elige el K donde añadir un clúster más ya no reduce el WCSS de forma significativa.

```python
from sklearn.cluster import KMeans

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Número de clústeres (K)')
plt.ylabel('WCSS')
plt.title('Método del Codo')
plt.show()
```

### - Coeficiente de Silueta

Mide simultáneamente la **cohesión interna** de cada cluster y la **separación** respecto al resto. Acotado entre -1 y 1.

| Valor | Interpretación |
|---|---|
| **≈ 1** | El individuo está bien asignado: cerca de los suyos, lejos de los demás |
| **≈ 0** | El individuo está en la frontera entre dos clusters |
| **≈ -1** | El individuo se parece más al cluster vecino que al suyo — mala asignación |

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"Coeficiente de Silueta: {score:.4f}")
```
