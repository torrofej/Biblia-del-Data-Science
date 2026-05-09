# Regresión en Machine Learning

---

## Índice

- [1. ¿Qué es la Regresión?](#1-qué-es-la-regresión)
- [2. Regresión Lineal Simple](#2-regresión-lineal-simple)
  - [2.1. Parámetros del modelo](#21-parámetros-del-modelo)
  - [2.2. Entrenamiento: función de coste](#22-entrenamiento-función-de-coste)
  - [2.3. Solución analítica (Mínimos Cuadrados)](#23-solución-analítica-mínimos-cuadrados)
  - [2.4. Implementación con scikit-learn](#24-implementación-con-scikit-learn)
- [3. Regresión Lineal Multivariable](#3-regresión-lineal-multivariable)
- [4. Métricas de Evaluación](#4-métricas-de-evaluación)
- [5. Regresión No Lineal](#5-regresión-no-lineal)
  - [5.1. Transformaciones simples](#51-transformaciones-simples)
  - [5.2. Expansión polinómica](#52-expansión-polinómica)
  - [5.3. Métodos Kernel](#53-métodos-kernel)
  - [5.4. Support Vector Machines (SVM)](#54-support-vector-machines-svm)
  - [5.5. Procesos Gaussianos (GP)](#55-procesos-gaussianos-gp)

---

## 1. ¿Qué es la Regresión?

La regresión es una técnica supervisada que modela la **relación entre una o varias variables independientes (features) y una variable dependiente continua (target)**.

El objetivo es aprender una función $f$ tal que:

$$\hat{y} = f(x_1, x_2, \dots, x_D)$$

donde $\hat{y}$ es el valor predicho. A diferencia de la clasificación, la salida es un **número real**, no una categoría.

**Ejemplos típicos**: precio de una vivienda, demanda eléctrica, temperatura futura.

---

## 2. Regresión Lineal Simple

La forma más básica asume que la relación entre la variable de entrada $x$ y la salida $y$ es lineal:

$$\hat{y} = b + w \cdot x$$

### 2.1. Parámetros del modelo

- **$b$ (bias / término independiente)**: desplazamiento vertical de la recta (punto de corte con el eje $y$). No multiplica a ninguna variable.
- **$w$ (weight / pendiente)**: tasa de variación de la salida respecto a la entrada. Indica cuánto cambia $\hat{y}$ por cada unidad de $x$.

### 2.2. Entrenamiento: función de coste

El aprendizaje automático consiste en encontrar los valores de $b$ y $\mathbf{w}$ que **minimizan el error** sobre los datos de entrenamiento.

El error individual para una observación es el **error cuadrático**:

$$\mathcal{L} = (y - \hat{y})^2$$

Se eleva al cuadrado para que siempre sea positivo. Sumando sobre todos los $N$ ejemplos:

$$\mathcal{L}(b, \mathbf{w}) = \sum_{n=1}^{N} (y_n - \hat{y}_n)^2$$

El objetivo es encontrar los parámetros óptimos $b^*$ y $\mathbf{w}^*$ que minimizan esta función de coste:

$$b^*, \mathbf{w}^* = \underset{b,\, \mathbf{w}}{\arg\min} \; \mathcal{L}(b, \mathbf{w})$$

La solución se obtiene derivando e igualando a cero:

$$\frac{d\mathcal{L}}{d\mathbf{w}} = 0$$

### 2.3. Solución analítica (Mínimos Cuadrados)

La solución cerrada de Mínimos Cuadrados (Least Squares) es:

$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$$

donde $\mathbf{X}$ es la matriz de features ($N \times D$, con una columna de unos para absorber $b$) e $\mathbf{Y}$ el vector de targets ($N \times 1$). **En la práctica no hay que implementar esto a mano**: scikit-learn lo gestiona internamente.

### 2.4. Implementación con scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print(f"Bias (b):    {modelo.intercept_:.4f}")
print(f"Pesos (w):   {modelo.coef_}")
```

---

## 3. Regresión Lineal Multivariable

Cuando hay $D$ variables de entrada, el modelo se extiende a:

$$\hat{y} = b + w_1 x_1 + w_2 x_2 + \dots + w_D x_D$$

Cada peso $w_i$ indica **cuánto cambia la salida por cada unidad de la variable $x_i$**, manteniendo el resto constantes.

**Ejemplo — precio de vivienda**:

| Variable | Peso | Interpretación |
|---|---|---|
| $x_1$ = m² | $w_1 = 3$ | +3.000 € por cada m² adicional |
| $x_2$ = distancia al centro (km) | $w_2 = -10$ | −10.000 € por cada km de distancia |

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

# X tiene múltiples columnas (features)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Asociar coeficientes a nombres de columnas
coefs = pd.Series(modelo.coef_, index=X_train.columns)
print(coefs.sort_values())
```

> ⚠️ Con más de 2 variables ya no es posible visualizar la regresión directamente. Se usan gráficos de **residuos** o representaciones parciales (una variable vs. predicción, con el resto fijo).

---

## 4. Métricas de Evaluación

Lo fundamental no es solo conocer las métricas, sino **saber cuándo usar cada una**.

### Error Absoluto Medio (MAE)

$$MAE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{n}$$

Robusto ante valores atípicos. Recomendado cuando el dataset contiene anomalías.

### Error Medio Cuadrático (MSE)

$$MSE = \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}$$

Penaliza errores grandes más que el MAE. Es diferenciable, lo que lo hace útil durante el entrenamiento.

### Raíz del Error Medio Cuadrático (RMSE)

$$RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}$$

Igual que MSE pero **mantiene las unidades originales** del problema. El más interpretable en la práctica.

### Porcentaje de Error Absoluto Medio (MAPE)

$$MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

Expresa el error en porcentaje. Útil para comparar modelos sobre variables con distintas escalas.

### Coeficiente de Determinación ($R^2$)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Indica qué **porcentaje de la variabilidad** de los datos explica el modelo. $R^2 = 1$ es ajuste perfecto; $R^2 = 0$ equivale a predecir siempre la media. Especialmente útil para **comparar modelos** que predicen la misma variable.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
```

**Guía de uso rápido**:

| Métrica | Cuándo usarla |
|---|---|
| **RMSE** | Caso general; penaliza errores grandes; mantiene las unidades |
| **MAE** | Dataset con muchos outliers; interpretación directa |
| **MAPE** | Comparación entre variables de distinta escala |
| **R²** | Comparar distintos modelos sobre la misma variable objetivo |

---

## 5. Regresión No Lineal

Muchas relaciones reales no son lineales. El mapa de técnicas disponibles:

```
Regresión no lineal
├── Transformaciones simples      (log, exp, sin...)
├── Expansión polinómica
├── Métodos Kernel
│   ├── SVM (Support Vector Machines)
│   └── Procesos Gaussianos (GP)
└── Modelos basados en árboles
    ├── Árboles de regresión
    ├── Random Forest
    └── Gradient Boosting Machines
```

> Los modelos basados en árboles (Random Forest, GBM) se cubren en detalle en el módulo de Analítica Avanzada.

### 5.1. Transformaciones simples

Cuando se conoce (o intuye) la forma funcional que siguen los datos, se puede **transformar** la variable antes de aplicar regresión lineal:

$$g(x) = \log(x) \qquad g(x) = \exp(x) \qquad g(x) = \sin(x)$$

El flujo es: $x \xrightarrow{\text{transformación}} g(x) \xrightarrow{f_{b,w}} \hat{y}$

El modelo resultante sigue siendo lineal en los parámetros $b$ y $w$:

$$\hat{y} = b + w \cdot g(x)$$

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Ejemplo: relación logarítmica
X_log = np.log(X)   # transformar la feature
modelo = LinearRegression()
modelo.fit(X_log, y)
```

### 5.2. Expansión polinómica

Se amplía la matriz de datos con potencias de las variables originales, permitiendo capturar **curvaturas** sin salir del marco lineal:

$$\hat{y} = b + w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4 + w_5 x^5$$

El **grado del polinomio** es un hiperparámetro a validar. A mayor grado, mayor riesgo de sobreajuste.

El flujo es: $x \xrightarrow{\text{expansión}} \tilde{x} \xrightarrow{f_{b,w}} \hat{y}$

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('lr',   LinearRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

> ⚠️ Siempre usar `Pipeline` para evitar data leakage al transformar train y test por separado.

### 5.3. Métodos Kernel

Un **kernel** es una función que mide la **similitud** entre dos observaciones $x$ y $x'$:

$$k(x, x')$$

Al aplicar un kernel, cada observación se transforma en su vector de distancias/similitudes con el resto del dataset. El flujo es: $X \xrightarrow{k(x,x')} \xrightarrow{f_{b,w}} Y$

**Tipos de kernel habituales**:

| Kernel | Fórmula | Uso |
|---|---|---|
| **Squared Exponential (SE / RBF)** | $\sigma_f^2 \exp\!\left(-\frac{(x-x')^2}{2l^2}\right)$ | Relaciones suaves y continuas |
| **Periódico (Per)** | $\sigma_f^2 \exp\!\left(-\frac{2}{l^2}\sin^2\!\left(\pi\frac{x-x'}{p}\right)\right)$ | Datos con estacionalidad |
| **Lineal (Lin)** | $\sigma_f^2(x - c)(x' - c)$ | Equivalente a regresión lineal |

Cada tipo de kernel introduce sus propios **hiperparámetros** a validar (longitud de escala $l$, varianza $\sigma_f^2$, etc.).

### 5.4. Support Vector Machines (SVM)

SVM busca el **hiperplano** que mejor modela la tendencia de los datos, maximizando el margen entre las bandas paralelas (vectores de soporte) que envuelven la nube de puntos.

**Función objetivo**:

$$\max \frac{1}{2}|\omega|^2 + C \sum_{i=1}^{N}(\xi_i + \xi_i^*)$$

- $\omega$: magnitud del hiperplano.
- $C$: constante que equilibra la regularidad del modelo frente al error cometido (hiperparámetro clave).
- $\xi$: variables de holgura que controlan el error al aproximar los vectores de soporte.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=1.0, epsilon=0.1))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Eficaz en espacios de alta dimensionalidad | Poco eficiente con datasets grandes |
| Gestión eficiente de memoria | Elección del kernel puede ser difícil |
| Flexible (múltiples kernels disponibles) | |
| Robusto frente a outliers | |

### 5.5. Procesos Gaussianos (GP)

Un Proceso Gaussiano es un modelo **no paramétrico** y bayesiano que predice **distribuciones completas sobre funciones**, no solo valores puntuales. Esto le permite cuantificar la **incertidumbre** de cada predicción.

Formalmente, $f(\boldsymbol{x})$ se modela como:

$$f(\boldsymbol{x}) \sim \mathcal{GP}\!\left(\mathbf{0},\; k(\boldsymbol{x}, \boldsymbol{x}')\right)$$

La distribución conjunta de observaciones y predicciones es:

$$\begin{bmatrix} \boldsymbol{y} \\ \boldsymbol{f}_* \end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\; \begin{bmatrix} K(\mathbf{X}, \mathbf{X}) + \sigma^2 \mathbf{I} & K(\mathbf{X}, \mathbf{X}_*) \\ K(\mathbf{X}_*, \mathbf{X}) & K(\mathbf{X}_*, \mathbf{X}_*) \end{bmatrix}\right)$$

La media y covarianza de las predicciones son:

$$\mathbb{E}[\boldsymbol{f}_*] = K(\mathbf{X}_*, \mathbf{X})\left[K(\mathbf{X}, \mathbf{X}) + \sigma^2 \mathbf{I}\right]^{-1} \boldsymbol{y}$$

$$\text{Cov}[\boldsymbol{f}_*] = K(\mathbf{X}_*, \mathbf{X}_*) - K(\mathbf{X}_*, \mathbf{X})\left[K(\mathbf{X}, \mathbf{X}) + \sigma^2 \mathbf{I}\right]^{-1} K(\mathbf{X}, \mathbf{X}_*)$$

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp.fit(X_train, y_train)

# Predicción con intervalo de confianza
y_pred, y_std = gp.predict(X_test, return_std=True)
```

**Ventajas y desventajas**:

| Ventajas | Desventajas |
|---|---|
| Proporciona intervalos de confianza en la predicción | Escala mal con datasets grandes ($O(N^3)$) |
| Muy flexible y no paramétrico | El kernel (función de covarianza) es difícil de elegir |
| Ideal cuando los datos son escasos pero críticos (medicina, sensores) | Los resultados son sensibles al ajuste del kernel |
| No requiere asumir la forma funcional de la relación | |
