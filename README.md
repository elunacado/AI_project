# Detection of poker hands
Esta es mi entrega individual del proyecto de IA para el profesor Benjamin Valdés Aguirre, el proyecto esta ordenado de la siguiente manera

* dataset
    * test
        test.data <- Dataset de test sin preprocesar
    train.data <- Dataset de train sin preprocesar
    validation.data <- Dataset de validation sin preprocesar
    * preprocesamiento <- Carpeta donde se realiza el preprocesamiento de los datos
    * construirValidation.py <- codigo para la generacion del split de validation (no viene uno incluido en el dataset original)
    * normalizacionsDelDataset.py <- noramlizacion del dataset mediante las tecnicas de Undersampling, Oversampling, One-Hot Encoding y Min-Max Scaler
    * preprocesamiento.py <- Archivo donde se generan los datasets preprocesados utulizando las funciones previamente establecidas
    * preprocesamiento_test.data -> el split de test preprocesado
    * preprocesamiento_trainning.data -> el split de trainning preprocesado
    * preprocesamiento_validation.data -> el split de validation normalizado
    * preprocesamiento.md -> la documentacion de la carpeta de preprocesamiento

* modelo
    * baseline: Definir una linea base utilizando el algoritmo de clasificacion Naive-Bayes
        * baselineNaiveBayes.ipynb <- codigo donde se calcula la linea base utilizando el algoritmo de clasificacion Naive-Bayes
        * image.png <- Imagen de la matriz de confusion (se utilizara en el .md)
        * Baseline.md <- Documentacion de la linea base

    * mlp_raw: Modelo de regresion utilizando los datos originales del dataset
        * MLP_raw.ipynb <- codigo donde se ejecuta el entrenamiento y exportacion del modelo MLP 
        * modelo_poker_raw.keras <- modelo exportado
        * pruebaDelMlp_raw.ipynb <- codigo para realizar queries al modelo
        * encoder.pkl <- codificador para escalar el input del usuario de acuerdo a los requisitos de preprocesamiento del modelo (One-Hot-Encoder)
        * scaler.pkl <- codificador para escalar el input del usuario de acuerdo a los requisitos de preprocesamiento del modelo (Min-Max Scaler)
        * image.png <- Imagen de la matriz de confusion (se utilizara en el .md)
        * MLP_raw.md <- la documentacion de la carpeta de mlp raw

* resultados
    * MLP_raw_Epoch10
        * output.txt <- Output del accuracy, recall y F1 tras correr el entrenamiento del MLP sin tener los datos preprocesados
    * Naive_Bayes

* README.md <- **Estas aqui**
