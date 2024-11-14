from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO

app = FastAPI()

class ProcesarArchivoService:
    @staticmethod
    def cargar_datos(archivo: BytesIO) -> pd.DataFrame:
        # Cargar el archivo en un DataFrame
        try:
            # Intentamos leer el archivo según su tipo
            df = pd.read_excel(archivo, engine='openpyxl')  # Usar openpyxl para archivos .xlsx y .xlsm
        except Exception as e:
            raise ValueError(f"Error al leer el archivo Excel: {str(e)}")
        
        # Limpieza de datos nulos
        df.fillna('', inplace=True)  # Rellena nulos con cadenas vacías
        
        # Asegurarse de que los tipos de datos sean compatibles
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(str)  # Convierte a cadena si la columna es float64
        
        # Convertir columnas datetime a string
        for col in df.select_dtypes(include=['datetime']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')  # Formato de fecha y hora en string

        return df

# Endpoint de FastAPI
@app.post("/cargar-datos/", response_description="Carga de datos desde archivos Excel o CSV")
async def cargar_datos(archivo: UploadFile = File(...), limite: int = 100):
    try:
        # Leemos el archivo cargado
        contenido = await archivo.read()
        archivo_bytes = BytesIO(contenido)
        
        # Procesamos el archivo usando la clase ProcesarArchivoService
        df = ProcesarArchivoService.cargar_datos(archivo_bytes)
        
        # Devolvemos las primeras filas (limitadas por el parámetro limite)
        return JSONResponse(content={"datos": df.head(limite).to_dict(orient="records")})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")
