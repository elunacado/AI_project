# 🃏 Detection of Poker Hands

Esta es mi entrega individual del proyecto de Inteligencia Artificial para el profesor **Benjamin Valdés Aguirre**.

**Poker Hand Classification**  
https://www.kaggle.com/datasets/dysphorfia/poker-hand-classification

El proyecto está organizado de la siguiente manera:

---

## 📁 Estructura del proyecto

### 🔹 dataset

Contiene los datos originales sin preprocesar:

* **test/**

  * `test.data` → Dataset de prueba sin preprocesar
* `train.data` → Dataset de entrenamiento sin preprocesar
* `validation.data` → Dataset de validación sin preprocesar

---

### 🔹 preprocesamiento

Carpeta donde se realiza todo el procesamiento de los datos:

* `construirValidation.py` → Código para generar el split de validación (el dataset original no lo incluye)
* `normalizacionDelDataset.py` → Normalización del dataset mediante:

  * Undersampling
  * Oversampling
  * One-Hot Encoding
  * Min-Max Scaler
* `preprocesamiento.py` → Script principal para generar los datasets preprocesados
* `preprocesamiento_test.data` → Dataset de prueba preprocesado
* `preprocesamiento_training.data` → Dataset de entrenamiento preprocesado
* `preprocesamiento_validation.data` → Dataset de validación preprocesado
* `preprocesamiento.md` → Documentación del proceso de preprocesamiento

---

### 🔹 modelo

#### 📌 baseline

Definición de una línea base utilizando el algoritmo de clasificación **Naive Bayes**:

* `baselineNaiveBayes.ipynb` → Código para calcular la línea base
* `image.png` → Imagen de la matriz de confusión
* `Baseline.md` → Documentación del baseline

---

#### 📌 mlp_raw

Modelo de red neuronal (MLP) entrenado con los datos originales (sin preprocesamiento):

* `MLP_raw.ipynb` → Entrenamiento y exportación del modelo
* `modelo_poker_raw.keras` → Modelo entrenado exportado
* `pruebaDelMlp_raw.ipynb` → Pruebas y consultas al modelo
* `encoder.pkl` → Codificador One-Hot para procesar inputs
* `scaler.pkl` → Escalador Min-Max para normalización
* `image.png` → Matriz de confusión
* `MLP_raw.md` → Documentación del modelo

---

### 🔹 resultados

* **MLP_raw_Epoch10**

  * `output.txt` → Resultados de accuracy, recall y F1 del MLP sin preprocesamiento

* **Naive_Bayes**

  * Resultados del modelo baseline

---

## 📍 Ethan Luna

Entrega individual — Proyecto de Inteligencia Artificial
