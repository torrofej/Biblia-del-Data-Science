# Clasificación en Machine Learning

---

## Índice

- [Clasificación en Machine Learning](#clasificación-en-machine-learning)
  - [Índice](#índice)
  - [6. ¿Qué es la Clasificación?](#6-qué-es-la-clasificación)
  - [7. Tipos de Clasificación](#7-tipos-de-clasificación)
  - [8. Encoding de Categorías (variable objetivo)](#8-encoding-de-categorías-variable-objetivo)
  - [9. Métricas de Evaluación](#9-métricas-de-evaluación)
    - [9.1. Conceptos base: TP, TN, FP, FN](#91-conceptos-base-tp-tn-fp-fn)
    - [9.2. Métricas derivadas](#92-métricas-derivadas)
    - [9.3. Curva ROC y AUC](#93-curva-roc-y-auc)
    - [9.4. Binary Cross Entropy (BCE)](#94-binary-cross-entropy-bce)
  - [10. Técnicas de Clasificación](#10-técnicas-de-clasificación)
  - [11. Regresión Logística](#11-regresión-logística)
    - [11.1. Función sigmoide](#111-función-sigmoide)
    - [11.2. Entrenamiento: Descenso por Gradiente (SGD)](#112-entrenamiento-descenso-por-gradiente-sgd)
    - [11.3. Implementación con scikit-learn](#113-implementación-con-scikit-learn)
  - [12. Naïve Bayes](#12-naïve-bayes)
    - [12.1. Teorema de Bayes](#121-teorema-de-bayes)
    - [12.2. Implementación con scikit-learn](#122-implementación-con-scikit-learn)

---

## 6. ¿Qué es la Clasificación?

La clasificación es una técnica de **aprendizaje supervisado** que consiste en predecir una **etiqueta (categoría)** para unos datos de entrada. A diferencia de la regresión, la salida no es un número continuo sino una clase discreta.

**Casos de uso habituales**:

| Tipo | Ejemplos |
|---|---|
| **Binaria** (2 clases) | Detección de spam, fraude, diagnóstico médico |
| **Multiclase** (>2 clases) | Detección de objetos, análisis de sentimiento, segmentación |

La clave para identificar un problema de clasificación es que la **variable objetivo toma valores de un conjunto finito de categorías** — no valores continuos.

---

## 7. Tipos de Clasificación

**Clasificación binaria** — la salida es una de dos clases (positivo/negativo, spam/no-spam, 0/1).

**Clasificación multiclase** — la salida es una de $K > 2$ clases mutuamente excluyentes. Cada observación pertenece exactamente a una clase.

**Clasificación multi-etiqueta** — cada observación puede pertenecer simultáneamente a **varias clases**. Por ejemplo, una imagen puede contener un gato *y* un pájaro a la vez. Las etiquetas se representan como un vector binario: `[1, 1, 0]` indica que pertenece a las clases 1 y 2 pero no a la 3.

```
Binaria:        [0] o [1]
Multiclase:     [1, 0, 0] o [0, 1, 0] o [0, 0, 1]
Multi-etiqueta: [1, 1, 0] o [0, 1, 1] o [1, 0, 1]
```

---

## 8. Encoding de Categorías (variable objetivo)

Antes de entrenar un modelo, las etiquetas de texto deben convertirse en valores numéricos. Hay tres estrategias principales:

**One-hot encoding** — crea una columna binaria por cada categoría. Recomendado para variables **nominales** (sin orden).

```python
import pandas as pd

df = pd.get_dummies(df, columns=['nivel_ahorros'])
# Resultado: columnas 'nivel_ahorros_Alto', '_Medio', '_Bajo'
```

**Label encoding** — asigna un entero arbitrario a cada categoría (Alto=1, Medio=0, Bajo=2). Útil para árboles de decisión, pero introduce **orden artificial** en algoritmos lineales.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['nivel_cod'] = le.fit_transform(df['nivel_ahorros'])
```

**Ordinal encoding** — asigna enteros respetando un **orden explícito** (Bajo=0, Medio=1, Alto=2). Solo usar cuando la variable tiene orden natural y ese orden es relevante.

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Bajo', 'Medio', 'Alto']])
df[['nivel_cod']] = oe.fit_transform(df[['nivel_ahorros']])
```

> ⚠️ Nunca usar Label Encoding con variables nominales en modelos lineales o de distancias (KNN, SVM): el modelo interpretará que "Bajo=2 > Medio=1" tiene significado matemático.

---

## 9. Métricas de Evaluación

### 9.1. Conceptos base: TP, TN, FP, FN

Todo clasificador binario produce cuatro tipos de resultados:

| Símbolo | Nombre | Descripción |
|---|---|---|
| **TP** | True Positive | Real=Positivo, predicho=Positivo ✓ |
| **TN** | True Negative | Real=Negativo, predicho=Negativo ✓ |
| **FP** | False Positive | Real=Negativo, predicho=Positivo ✗ |
| **FN** | False Negative | Real=Positivo, predicho=Negativo ✗ |

La **matriz de confusión** muestra estos cuatro valores de forma tabular y es el punto de partida para calcular todas las métricas.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

### 9.2. Métricas derivadas

**Accuracy** — proporción de predicciones correctas sobre el total:

$$Acc = \frac{TP + TN}{TP + TN + FP + FN}$$

> ⚠️ Con clases desbalanceadas, la accuracy puede ser engañosa (un modelo que siempre predice "no fraude" puede tener 99% de accuracy).

**Precision** — de todos los predichos como positivos, ¿cuántos lo son realmente?

$$Precision = \frac{TP}{TP + FP}$$

**Recall (Exhaustividad)** — de todos los positivos reales, ¿cuántos detecta el modelo?

$$Recall = \frac{TP}{TP + FN}$$

**Especificidad** — de todos los negativos reales, ¿cuántos identifica correctamente?

$$Especificidad = \frac{TN}{TN + FP}$$

**F1-Score** — media armónica de Precision y Recall. Útil cuando hay desbalanceo de clases:

$$F1 = 2 \cdot \frac{Recall \cdot Precision}{Recall + Precision}$$

**Guía de cuándo usar cada métrica**:

| Métrica | Cuándo priorizarla |
|---|---|
| **Accuracy** | Clases balanceadas, visión general |
| **Precision** | Coste alto de falsos positivos (ej. spam: no quieres perder emails legítimos) |
| **Recall** | Coste alto de falsos negativos (ej. detección de cáncer: no quieres perder casos) |
| **F1** | Desbalanceo de clases; cuando importan tanto FP como FN |

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Neg', 'Pos']))
```

### 9.3. Curva ROC y AUC

La **curva ROC** representa la tasa de verdaderos positivos (Recall) frente a la tasa de falsos positivos (1 - Especificidad) para todos los umbrales de clasificación posibles.

El **AUC** (Area Under the Curve) resume la curva en un único número:

- $AUC = 1.0$ → clasificador perfecto.
- $AUC = 0.5$ → equivale a clasificar aleatoriamente (línea diagonal).

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = modelo.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### 9.4. Binary Cross Entropy (BCE)

La BCE es la **función de coste** utilizada para entrenar clasificadores binarios. Mide cuánto difieren las predicciones probabilísticas del modelo de las etiquetas reales y **penaliza fuertemente la confianza en predicciones incorrectas**.

$$BCE = -\left(y \cdot \log f(x) + (1 - y) \cdot \log(1 - f(x))\right)$$

Para todos los ejemplos del conjunto de entrenamiento:

$$\mathcal{L}(b, \mathbf{w}) = -\frac{1}{N} \sum_{n=1}^{N} \left(y_n \cdot \log f(x_n) + (1 - y_n) \cdot \log(1 - f(x_n))\right)$$

Desglosando por caso:
- Si $y = 1$: $BCE = -\log f(x)$ — penaliza si el modelo da probabilidad baja al positivo real.
- Si $y = 0$: $BCE = -\log(1 - f(x))$ — penaliza si el modelo da probabilidad alta al negativo real.

---

## 10. Técnicas de Clasificación

El mapa de técnicas disponibles sigue la misma estructura que en regresión:

```
Clasificación
├── Lineal
│   ├── Regresión Logística
│   └── Naïve Bayes
└── No lineal
    ├── Transformaciones simples / polinómicas
    ├── Métodos Kernel (SVM, Gaussian Processes)
    ├── Modelos basados en árboles
    │   ├── Árboles de decisión
    │   ├── Random Forest
    │   └── Gradient Boosting Machines
    └── Redes neuronales
```

> Los métodos Kernel (SVM para clasificación) y los modelos basados en árboles comparten la misma lógica que en regresión — cambia la función de coste y la salida final, pero no la estructura del algoritmo.

---

## 11. Regresión Logística

A pesar del nombre, es un **algoritmo de clasificación**. Parte del mismo modelo lineal de la regresión pero añade una función que transforma la salida en una **probabilidad entre 0 y 1**.

La idea base es que un clasificador lineal predice positivo cuando $f(x) \geq 0$ y negativo cuando $f(x) < 0$, usando el hiperplano $f(x) = b + w_1 x_1 + \dots + w_D x_D$ como frontera de decisión.

### 11.1. Función sigmoide

Para obtener probabilidades, se aplica la función **sigmoide** $\sigma$ a la salida lineal:

$$f(x) = \sigma(b + w \cdot x) = \frac{1}{1 + \exp(-(b + w \cdot x))}$$

La sigmoide compacta cualquier número real al rango $(0, 1)$, interpretable directamente como probabilidad de pertenecer a la clase positiva.

- $f(x) \geq 0.5$ → clase positiva
- $f(x) < 0.5$ → clase negativa

### 11.2. Entrenamiento: Descenso por Gradiente (SGD)

La sigmoide elimina la posibilidad de una solución cerrada (como la de Mínimos Cuadrados en regresión lineal). En su lugar se usa **Descenso por Gradiente Estocástico (SGD)**, que actualiza iterativamente los parámetros:

$$b_{t+1},\, \mathbf{w}_{t+1} = b_t,\, \mathbf{w}_t - \eta \, \nabla_{b,w} \mathcal{L}(b, \mathbf{w};\, x_i, y_i)$$

donde $\eta$ es el **learning rate** (tasa de aprendizaje), el hiperparámetro más crítico del proceso:

| Learning rate | Efecto |
|---|---|
| **Demasiado bajo** | Convergencia lenta; necesita muchas iteraciones |
| **Adecuado** | Converge rápidamente al mínimo |
| **Demasiado alto** | Actualizaciones divergentes; el coste no baja |

### 11.3. Implementación con scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

modelo = LogisticRegression(max_iter=1000, random_state=42) # Sigmoide por defecto, para SGD usar parámetro solver = 'saga'
modelo.fit(X_train, y_train)

y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]  # probabilidad clase positiva

print(classification_report(y_test, y_pred))
print(f"Pesos (w): {modelo.coef_}")
print(f"Bias  (b): {modelo.intercept_}")
```

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Simple y rápido de entrenar | Modelo lineal: poca flexibilidad |
| Coeficientes interpretables (importancia de cada variable) | No captura relaciones entre variables de entrada |
| Produce probabilidades, no solo clases | Sensible a outliers |

---

## 12. Naïve Bayes

Clasificador **probabilístico** basado en el teorema de Bayes. Asume que todas las features son **independientes entre sí** dado el valor de la clase (de ahí el "naïve" / ingenuo).

### 12.1. Teorema de Bayes

El teorema de Bayes expresa la probabilidad posterior de la clase $y$ dado el documento (o conjunto de features) $D$:

$$P(y|D) = \frac{P(D|y) \cdot P(y)}{P(D)}$$

Donde:
- $P(y|D)$ — **posterior**: probabilidad de la clase $y$ dado $D$ (lo que queremos).
- $P(D|y)$ — **likelihood**: probabilidad de observar $D$ si la clase es $y$.
- $P(y) = \dfrac{n_y}{N}$ — **prior**: frecuencia de la clase $y$ en el dataset de entrenamiento.
- $P(D)$ — **evidencia**: constante normalizadora (igual para todas las clases, se ignora al comparar).

La predicción elige la clase con mayor probabilidad posterior:

$$\hat{y} = \underset{y}{\arg\max} \; \frac{P(D|y) \cdot P(y)}{P(D)}$$

**Elección del likelihood** $P(D|y)$:
- Features **discretas** (ej. frecuencia de palabras): se cuentan las frecuencias → **Multinomial NB**.
- Features **continuas**: se modela cada feature con una Gaussiana → **Gaussian NB**.

### 12.2. Implementación con scikit-learn

```python
from sklearn.naive_bayes import GaussianNB

# Para features continuas
modelo = GaussianNB()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)
```

**Ejemplo con texto (clasificación de spam)**:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),   # texto → frecuencias de palabras
    ('clf',        MultinomialNB())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
```

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Muy rápido de entrenar, incluso con muchas features | Asume independencia entre features (raramente cierto) |
| Funciona bien con datos escasos | Puede construir un modelo que no se ajuste bien a los datos |
| Fácil de integrar conocimiento previo (prior) | Requiere elegir likelihood y prior, lo que puede ser complejo |
| Escala bien con alta dimensionalidad (ej. NLP) | |
