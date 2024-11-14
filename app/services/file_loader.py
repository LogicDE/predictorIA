import pandas as pd

def leer_excel(file_path: str):
    try:
        datos = pd.read_excel(file_path)
        return datos
    except Exception as e:
        print(f"Error al leer el archivo Excel: {e}")
        return None
