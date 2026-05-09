# Clasificación en Machine Learning (II)

---

## Índice

- [13. Separación No Lineal](#13-separación-no-lineal)
- [14. Transformaciones Simples para Clasificación](#14-transformaciones-simples-para-clasificación)
- [15. Expansión Polinómica para Clasificación](#15-expansión-polinómica-para-clasificación)
- [16. Support Vector Machines (SVM)](#16-support-vector-machines-svm)
  - [16.1. Margen máximo y optimización](#161-margen-máximo-y-optimización)
  - [16.2. SVM con datos no separables (soft margin)](#162-svm-con-datos-no-separables-soft-margin)
  - [16.3. SVM con kernels](#163-svm-con-kernels)
  - [16.4. Implementación con scikit-learn](#164-implementación-con-scikit-learn)
- [17. K-Nearest Neighbors (KNN)](#17-k-nearest-neighbors-knn)
- [18. Gaussian Process Classifier (GPC)](#18-gaussian-process-classifier-gpc)
- [19. Árboles de Decisión](#19-árboles-de-decisión)
  - [19.1. Índice Gini vs Entropía](#191-índice-gini-vs-entropía)
  - [19.2. Implementación con scikit-learn](#192-implementación-con-scikit-learn)
- [20. Random Forest](#20-random-forest)
  - [20.1. ¿Qué es un ensemble?](#201-qué-es-un-ensemble)
  - [20.2. Implementación con scikit-learn](#202-implementación-con-scikit-learn)
- [21. Clasificación Multiclase: OvR y OvO](#21-clasificación-multiclase-ovr-y-ovo)

---

## 13. Separación No Lineal

Muchos problemas reales no son linealmente separables: no existe un único hiperplano que divida perfectamente las clases. Esto puede ocurrir con una o varias variables de entrada.

Con una variable, la clase positiva puede ocupar **dos regiones distintas** (ej. valores bajos y altos de $x$, con la clase negativa en el centro), lo que un modelo lineal no puede capturar. Con dos variables, las clases pueden estar dispuestas en patrones circulares, espirales u otras formas complejas que requieren **fronteras de decisión curvas**.

La solución pasa por aplicar transformaciones a las features antes de entrenar el clasificador lineal.

---

## 14. Transformaciones Simples para Clasificación

Se aplica la misma idea que en regresión: transformar las variables de entrada con funciones conocidas ($\log$, $\exp$, $\sin$, coordenadas polares, etc.) antes de pasar el resultado a la sigmoide.

**Ejemplo 1 — dos regiones con $x^2$**: si la clase positiva ocupa los extremos de $x$ y la negativa el centro, añadir $x^2$ como feature captura esa curvatura:

$$f(x) = \sigma(b + w_1 x + w_2 x^2) \quad \text{Accuracy: 0.970 vs 0.600 lineal}$$

**Ejemplo 2 — datos periódicos con $\sin(x)$**: si la frontera oscila, una transformación sinusoidal puede modelarla:

$$f(x) = \sigma(b + w_1 \sin(x)) \quad \text{Accuracy: 0.850 vs 0.660 lineal}$$

**Ejemplo 3 — estructura circular**: cuando la clase positiva forma un círculo interior y la negativa un anillo, conviene transformar a **coordenadas polares**:

$$r(x_1, x_2) = \sqrt{x_1^2 + x_2^2} \qquad \alpha(x_1, x_2) = \arctan\!\left(\frac{x_2}{x_1}\right)$$

$$f(x_1, x_2) = \sigma\!\left(b + w_1 \cdot r(x_1, x_2) + w_2 \cdot \alpha(x_1, x_2)\right) \quad \text{Accuracy: 0.890 vs 0.507 lineal}$$

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Coordenadas polares como nuevas features
r     = np.sqrt(X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
alpha = np.arctan2(X[:, 1], X[:, 0]).reshape(-1, 1)
X_polar = np.hstack([r, alpha])

modelo = LogisticRegression()
modelo.fit(X_polar, y)
```

> ⚠️ Las transformaciones simples requieren que el analista **conozca o intuya** la forma funcional de los datos. Cuando no hay patrón claro, hay que recurrir a la expansión polinómica o a métodos kernel.

---

## 15. Expansión Polinómica para Clasificación

Cuando no hay un patrón claro que permita elegir una transformación manual, la expansión polinómica genera sistemáticamente todas las combinaciones de potencias de las features hasta un grado dado.

Para dos variables $x_1$ y $x_2$ con grado 3, el modelo pasa de:

$$f(x_1, x_2) = \sigma(b + w_1 x_1 + w_2 x_2) \quad \text{Accuracy: 0.863}$$

a incluir también términos cuadráticos y cúbicos:

$$f(x_1, x_2) = \sigma(b + w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2 + w_6 x_1^3 + \dots) \quad \text{Accuracy: 0.927}$$

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('clf',  LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

> ⚠️ A mayor grado, mayor riesgo de sobreajuste. Validar el grado con cross-validation.

---

## 16. Support Vector Machines (SVM)

SVM busca el hiperplano que **maximiza el margen** de separación entre las clases, en lugar de simplemente encontrar cualquier hiperplano que las separe.

### 16.1. Margen máximo y optimización

Entre todos los hiperplanos que separan las clases, el óptimo es el que deja la mayor distancia a los puntos más cercanos de cada clase (los **vectores soporte**). El margen viene dado por $\dfrac{1}{\|\mathbf{w}\|^2}$.

El problema de optimización se formula como:

$$\min_{\mathbf{w},\, b} \frac{1}{\|\mathbf{w}\|^2} \quad \text{sujeto a} \quad y_i \left(\mathbf{w}^\top \mathbf{x}_i + b\right) \geq 1 \quad \forall i$$

Solo los puntos que quedan **sobre el margen o dentro de él** se convierten en vectores soporte y determinan el hiperplano. El resto de los puntos no influyen.

Este problema se resuelve con el **Lagrangiano**, obteniendo el problema dual:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \quad \text{s.a.} \quad \sum_i \alpha_i y_i = 0,\; \alpha_i \geq 0 \; \forall i$$

### 16.2. SVM con datos no separables (soft margin)

Cuando las clases no son perfectamente separables, se introduce la variable de holgura $\xi_i \geq 0$, que mide cuánto viola cada punto el margen:

$$\min_{\mathbf{w},\, b,\, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{sujeto a} \quad y_i\left(\mathbf{w}^\top \mathbf{x}_i + b\right) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

El hiperparámetro **$C$** controla el equilibrio entre maximizar el margen y permitir errores: un $C$ grande penaliza más los errores (margen más estrecho, menos tolerancia); un $C$ pequeño permite más violaciones (margen más amplio).

### 16.3. SVM con kernels

Para fronteras de decisión complejas, SVM puede usar un **kernel** $K(x, x')$ que transforma implícitamente los datos a un espacio de mayor dimensión sin calcularlo explícitamente:

| Kernel | Fórmula |
|---|---|
| **Lineal** | $K(x, x') = x \cdot x'$ |
| **Polinómico** | $K(x, x') = (x \cdot x' + 1)^d$ |
| **RBF (Gaussiano)** | $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ |
| **Sigmoide** | $K(x, x') = \tanh(\alpha x \cdot x' + c)$ |

### 16.4. Implementación con scikit-learn

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),         # SVM es sensible a la escala
    ('svc',    SVC(kernel='rbf', C=1.0, probability=True))
])
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
```

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Eficaz en espacios de alta dimensionalidad | Poco eficiente con datasets grandes |
| Gestión eficiente de memoria | Elección del kernel puede ser difícil |
| Flexible gracias a la variedad de kernels | |
| Robusto frente a outliers | |

---

## 17. K-Nearest Neighbors (KNN)

KNN es un algoritmo **no paramétrico** y de **lazy learning** (no aprende un modelo explícito durante el entrenamiento; simplemente memoriza los datos). Para clasificar un nuevo punto, busca sus $K$ vecinos más cercanos y asigna la clase mayoritaria.

**Algoritmo paso a paso:**

1. Calcular la distancia euclídea del nuevo punto $\mathbf{x}_{new}$ a cada punto del dataset:

$$\text{dist}(\mathbf{x}_{new}, \mathbf{x}_i) = \|\mathbf{x}_{new} - \mathbf{x}_i\|_2$$

2. Seleccionar los $K$ puntos con menor distancia.

3. Asignar la clase más frecuente entre los $K$ vecinos:

$$\hat{y}_{new} = \text{moda}\left(\{y_i \mid i \in \text{K vecinos más cercanos}\}\right)$$

$K$ es el **hiperparámetro** clave: un $K$ pequeño produce fronteras muy irregulares (sobreajuste); un $K$ grande suaviza la frontera pero puede perder detalle local.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),   # KNN es muy sensible a la escala
    ('knn',    KNeighborsClassifier(n_neighbors=5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Muy fácil de entender e implementar | Lento en predicción con datasets grandes ($O(N)$ por consulta) |
| No requiere entrenamiento (lazy learning) | Muy sensible a la escala de las variables |
| Funciona bien en problemas no lineales | Rendimiento degradado con alta dimensionalidad |

---

## 18. Gaussian Process Classifier (GPC)

El GPC extiende el GP de regresión al problema de clasificación. Se introduce una **función latente** $f(\mathbf{x})$ no observada que sigue una distribución GP:

$$f(\mathbf{x}) \sim \mathcal{GP}(0,\, k(\mathbf{x}, \mathbf{x}'))$$

Esta función latente se transforma en probabilidad de clase 1 mediante la sigmoide:

$$\pi(\mathbf{x}) = \sigma(f(\mathbf{x})) = \frac{1}{1 + e^{-f(\mathbf{x})}}$$

La predicción requiere integrar sobre la distribución posterior de $f_*$, lo que no tiene solución cerrada y se resuelve con aproximaciones:

$$p(y_* = 1 \mid \mathbf{x}_*, \mathcal{D}) = \int \sigma(f_*)\, p(f_* \mid \mathbf{x}_*, \mathcal{D})\, df_*$$

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
gpc.fit(X_train, y_train)
y_pred  = gpc.predict(X_test)
y_proba = gpc.predict_proba(X_test)
```

Las ventajas y desventajas del GPC son las mismas que las del GP de regresión: proporciona incertidumbre en la predicción, es muy flexible, pero escala mal con $N$ grande y su rendimiento depende fuertemente del kernel elegido.

---

## 19. Árboles de Decisión

Un árbol de decisión divide recursivamente el espacio de features mediante **preguntas binarias** sobre el valor de cada variable (ej. "¿es `petal_length` < 2.45?"). Cada nodo interno aplica un umbral a una feature; las hojas contienen la clase predicha.

En cada división, el algoritmo elige el corte que **maximiza la pureza** de los nodos hijos. Las dos métricas de pureza más usadas son el Índice Gini y la Entropía.

### 19.1. Índice Gini vs Entropía

**Índice Gini** — mide la impureza de un nodo. A mayor Gini, menor pureza (más mezcla de clases):

$$Gini(t) = 1 - \sum_{i=1}^{n} p_i^2$$

**Entropía** — mide el desorden del sistema. A menor entropía, mayor orden (mayor pureza):

$$Entropia = -\sum_{i=1}^{n} p_i \cdot \log_2 p_i$$

donde $p_i$ es la probabilidad de que una muestra pertenezca a la clase $i$.

Ambas métricas producen resultados similares en la práctica. Gini es ligeramente más rápido de calcular; Entropía tiende a generar árboles más balanceados.

### 19.2. Implementación con scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

modelo = DecisionTreeClassifier(
    criterion='gini',   # o 'entropy'
    max_depth=5,        # limitar profundidad evita sobreajuste
    random_state=42
)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print(classification_report(y_test, y_pred))
print(export_text(modelo, feature_names=list(X_train.columns)))

# Importancia de variables
import pandas as pd
importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
print(importancias.sort_values(ascending=False))
```

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Fácil de interpretar y visualizar | Propenso al sobreajuste sin poda o límite de profundidad |
| Requiere poca preparación de datos | Alta varianza: pequeños cambios en datos → árbol muy diferente |
| Válido para clasificación y regresión | |

---

## 20. Random Forest

Random Forest es un método de **ensemble** que combina múltiples árboles de decisión entrenados sobre subconjuntos aleatorios de datos y features. La predicción final es la **clase mayoritaria** entre todos los árboles (majority voting).

### 20.1. ¿Qué es un ensemble?

Un ensemble usa múltiples modelos base para resolver un único problema y combina sus predicciones para obtener una solución más robusta que cualquier modelo individual. Existen tres variantes principales:

**Bagging** (Random Forest) — cada modelo base se entrena sobre un subconjunto aleatorio del dataset (con reemplazo). Las predicciones se promedian o votan. Reduce la varianza.

**Boosting** — cada modelo base se entrena corrigiendo los errores del anterior (el dataset de entrenamiento se modifica en función de las predicciones previas). Reduce el sesgo.

**Stacking** — se entrenan modelos base heterogéneos y un meta-modelo (meta-learner) aprende a combinar sus predicciones.

### 20.2. Implementación con scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

modelo = RandomForestClassifier(
    n_estimators=100,      # número de árboles
    criterion='gini',      # métrica de pureza
    max_depth=None,        # sin límite de profundidad por árbol
    n_jobs=-1,             # usar todos los procesadores disponibles
    class_weight='balanced',  # útil con clases desbalanceadas
    random_state=42
)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Importancia de variables
import pandas as pd
importancias = pd.Series(
    modelo.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print(importancias)
```

**Parámetros clave**:

| Parámetro | Descripción |
|---|---|
| `n_estimators` | Número de árboles del bosque |
| `criterion` | Métrica de pureza: `'gini'` o `'entropy'` |
| `max_depth` | Profundidad máxima de cada árbol |
| `n_jobs` | Procesadores en paralelo (`-1` = todos) |
| `class_weight` | Pesos por clase para datasets desbalanceados |

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Gran rendimiento y baja varianza | Alto coste computacional |
| Alta estabilidad y robustez | Alta correlación entre árboles con datasets pequeños o similares |
| Versatilidad (clasificación y regresión) | |
| Proporciona importancia de variables | |

---

## 21. Clasificación Multiclase: OvR y OvO

Muchos clasificadores binarios (Regresión Logística, SVM) pueden extenderse a $K > 2$ clases mediante dos estrategias:

**One-vs-Rest (OvR)** — se entrena un clasificador binario por cada clase: el clasificador de la clase $k$ aprende a distinguir "clase $k$" vs "todas las demás". Para $K$ clases se entrenan $K$ modelos. En la predicción se elige la clase cuyo clasificador da la mayor puntuación.

**One-vs-One (OvO)** — se entrena un clasificador por cada **par** de clases. Para $K$ clases se entrenan $\dfrac{K(K-1)}{2}$ modelos. OvO escala peor con $K$ pero cada clasificador trabaja con menos datos (solo las dos clases en juego), lo que puede ser ventajoso.

**Nativos** — algunos modelos como KNN, árboles de decisión y redes neuronales admiten multiclase de forma nativa sin necesidad de OvR ni OvO.

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

base_clf = LogisticRegression(max_iter=1000)

# One-vs-Rest
clf_ovr = OneVsRestClassifier(base_clf)
clf_ovr.fit(X_train, y_train)
print(f"OvR Accuracy: {accuracy_score(y_test, clf_ovr.predict(X_test)):.3f}")

# One-vs-One
clf_ovo = OneVsOneClassifier(base_clf)
clf_ovo.fit(X_train, y_train)
print(f"OvO Accuracy: {accuracy_score(y_test, clf_ovo.predict(X_test)):.3f}")
```

> 💡 En scikit-learn, `LogisticRegression` y `SVC` con `kernel='linear'` ya aplican OvR por defecto cuando `y` tiene más de dos clases. No es necesario envolverlos manualmente salvo que se quiera forzar OvO.
