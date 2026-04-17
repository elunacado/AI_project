import pandas as pd
from sklearn.model_selection import train_test_split

def dividir_dataset(input_file, train_file='train.data', validation_file='validation.data', 
                  test_size=0.2, random_state=42):
    
    # Cargar dataset
    df = pd.read_csv(input_file, header=None)
    
    # Dividir el dataset en entrenamiento y validación en una proporción de 80 - 20 tras haberlos revuelto,
    #  brindamos una semilla para poder reproducir los resultados
    train, validation = train_test_split(df, test_size=test_size, 
                                          random_state=random_state, shuffle=True)
    """
    #Guardar
    train.to_csv(train_file, index=False, header=None)
    validation.to_csv(validation_file, index=False, header=None)
    
    print(f"Training:   {len(train)} filas  -> {train_file}")
    print(f"Validación: {len(validation)} filas -> {validation_file}")
    """
    
    return train, validation