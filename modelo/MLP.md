**Se utiliza un MLP para la prediccion de los datos** 

Este script entrena y evalúa una red neuronal multicapa (MLP) usando Keras sobre el mismo dataset de manos de póker. A continuación se describe cada sección:

Se lee el archivo CSV preprocesado (preprocesamiento_trainning.data) sin encabezado. Las columnas se dividen en:
  - x: todas las columnas menos la última 
  - y: la última columna 

Las clases son 10 tipos de manos de póker numeradas del 0 al 9:
  0: High Card        5: Flush
  1: One Pair         6: Full House
  2: Two Pairs        7: Four of a Kind
  3: Three of a Kind  8: Straight Flush
  4: Straight         9: Royal Flush

También se calcula:
  - n_clases  : número de clases únicas en y (10).
  - n_features: número de columnas de entrada en x.

2. ARQUITECTURA DEL MODELO (MLP)
---------------------------------
Se construye una red neuronal secuencial con las siguientes capas:

  [Entrada]  → vector de tamaño n_features
      ↓
  [Dense 64, ReLU]  → primera capa oculta, 64 neuronas
      ↓
  [Dense 32, ReLU]  → segunda capa oculta, 32 neuronas
      ↓
  [Dense n_clases, Softmax] → capa de salida, una neurona por clase

  - ReLU (Rectified Linear Unit): activa la neurona solo si el valor es
    positivo; introduce no-linealidad sin saturar el gradiente.
  - Softmax: convierte los valores de salida en probabilidades que suman 1,
    una por cada clase.

modelo_tf.summary() imprime un resumen con el número de parámetros
entrenables por capa.

3. COMPILACIÓN
--------------
El modelo se compila con:
  - optimizer='adam'   : algoritmo de optimización adaptativo, ajusta la
                         tasa de aprendizaje automáticamente durante el
                         entrenamiento.
  - loss='sparse_categorical_crossentropy': función de pérdida para
                         clasificación multiclase cuando las etiquetas son
                         enteros (no one-hot encoded).
  - metrics=['accuracy']: métrica monitoreada durante el entrenamiento.

4. ENTRENAMIENTO
----------------
El modelo se entrena con modelo_tf.fit() usando los siguientes parámetros:
  - epochs=50         : el dataset completo se recorre 50 veces.
  - batch_size=32     : los pesos se actualizan cada 32 muestras.
  - validation_split=0.2 : el 20% de los datos se separa automáticamente
                         para validar el modelo en cada época, sin usarse
                         en el entrenamiento.
  - verbose=1         : muestra el progreso (loss y accuracy) por época.

El objeto 'history' guarda el historial de métricas por época, útil para
graficar curvas de aprendizaje.

5. PREDICCIÓN Y MÉTRICAS
--------------------------
modelo_tf.predict(x) devuelve una matriz de probabilidades (una fila por
muestra, una columna por clase). np.argmax(..., axis=1) selecciona el índice
de la clase con mayor probabilidad, convirtiéndolo en la etiqueta predicha.

Se imprimen tres métricas sobre el conjunto de entrenamiento completo:
  - Accuracy : proporción de predicciones correctas sobre el total.
  - Recall   : promedio macro del recall por clase.
  - F1 Score : promedio macro del F1, balance entre precisión y recall.

6. MATRIZ DE CONFUSIÓN
-----------------------
Se calcula y visualiza la matriz de confusión con seaborn (heatmap azul).
  - Eje Y (Real):     clase verdadera de cada muestra.
  - Eje X (Predicho): clase asignada por el modelo.

Las celdas en la diagonal son aciertos; las fuera de ella son errores.
Las etiquetas del eje X se rotan 45° para facilitar la lectura.

DIFERENCIAS CLAVE RESPECTO AL MODELO NAIVE BAYES
-------------------------------------------------
  - El MLP aprende representaciones internas no lineales; Naive Bayes asume
    distribuciones gaussianas fijas.
  - El MLP puede capturar relaciones entre features; Naive Bayes asume
    independencia condicional.
  - El MLP requiere más datos y tiempo de entrenamiento; Naive Bayes es
    casi instantáneo.
  - El uso de validation_split permite detectar sobreajuste (overfitting)
    durante el entrenamiento, algo que el código de Naive Bayes no hacía.


MLP:
https://scikit--learn-org.translate.goog/stable/modules/neural_networks_supervised.html?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc