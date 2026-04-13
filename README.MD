# Proyecto Individual de IA

Bienvenido a mi proyecto individual de Inteligencia Artificial para la clase del profesor Benjamin Valdez Aguirre.

Fecha de entrega: 13 de abril de 2026  

---

## Dataset

Se utilizó el siguiente dataset:

Poker Hand Classification  
https://www.kaggle.com/datasets/dysphoria/poker-hand-classification

---

## Objetivo de la entrega

Para esta entrega se solicitó:

- Seleccionar un dataset  
- Realizar la separación de los datos  
- Aplicar un proceso de preprocesamiento  
- Preparar el dataset para ser utilizado posteriormente en un modelo  

---

## Ejecución

El archivo principal de preprocesamiento es:

```bash
python preprocesamiento.py
```
    

   El preprocesamiento consta de lo siguiente:

    ```python
    import pandas as pd
    from normalizacionDelDataset import *
    from construirValidation import dividir_dataset
    from deNormalizar import deNormalizar
    ```

    * Importacion de pandas
    * Importacion de funciones de normalizacion
    * Importar el codigo para la separacion del trainning de validacion
    * Importar el codigo para la denormalizacion

    ```python
    training = '../dataset/train.data'
    OBJETIVO_PARA_ENTRENAR = {
        0: 100000,  # Nothing in hand
        1: 100000,  # One pair
        2: 30000,  # Two pairs
        3: 30000,  # Three of a kind
        4: 10000,  # Straight
        5: 10000,  # Flush
        6: 10000,  # Full house
        7: 1000,  # Four of a kind
        8: 100,  # Straight flush
        9: 100   # Royal flush
    }
    ```
    Posteriormente, se cargan los datos de entrenamiento junto con los objetivos de balanceo (reducción y aumento de datos), los cuales se utilizarán para aplicar técnicas de oversampling y undersampling con el fin de mejorar el balance del dataset.

    ```python
        df_trainning = pd.read_csv(training, header=None)
        df_trainning = quitarNull(df_trainning)
        df_oversampleado = oversamplearDataset(df_trainning, OBJETIVO_PARA_ENTRENAR)
        df_final = undersamplearDataset(df_oversampleado, OBJETIVO_PARA_ENTRENAR)
        df_final, scaler = normalizarDataset(df_final)
    ```
    El flujo de preprocesamiento aplicado al conjunto de datos consiste en varias etapas:

    1. Lectura del archivo CSV
    2. Eliminación de filas con valores nulos
    3. Aplicación de oversampling para las clases minoritarias (principalmente de la clase 3 en adelante)
    4. Aplicación de undersampling para las clases mayoritarias (de la 0 a la 2), siempre considerando los valores   definidos en OBJETIVO_PARA_ENTRENAR
    5. Normalización de los datos

    La normalización se realiza transformando los valores numéricos a un rango más manejable, por ejemplo, convirtiendo valores máximos como 4 y 13 a 1, lo que facilita el procesamiento por parte del modelo.

    ```python
        evaluacionDelBalanceoDelDataset(df_final)
        df_final.to_csv('preprocessed_train.data', index=False, header=None)

    ```
    Finalmente, se evalúa el balance del dataset para verificar la nueva distribución de clases y se guarda el conjunto preprocesado en un archivo .data.