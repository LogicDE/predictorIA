import pandas as pd
from pymongo import MongoClient
from app.config import settings
from io import BytesIO

class ProcesarArchivoService:
    
    @staticmethod
    def cargar_datos_xlsx(archivo: BytesIO) -> pd.DataFrame:
        try:
            # Leemos el archivo .xlsx usando openpyxl
            df = pd.read_excel(archivo, engine='openpyxl')  # Usamos openpyxl para archivos .xlsx y .xlsm

            # Limpieza de datos nulos
            df.fillna('', inplace=True)  # Rellena nulos con cadenas vacías

            # Asegurarse de que los tipos de datos sean compatibles
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(str)  # Convierte a cadena si la columna es float64

            # Convertir columnas datetime a string
            for col in df.select_dtypes(include=['datetime']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')  # Formato de fecha y hora en string

            return df
        except Exception as e:
            raise ValueError(f"Error al procesar el archivo .xlsx: {str(e)}")
    
    @staticmethod
    def cargar_datos_xls(archivo: BytesIO) -> pd.DataFrame:
        try:
            # Leemos el archivo .xls usando xlrd
            df = pd.read_excel(archivo, engine='xlrd')  # Usamos xlrd para archivos .xls

            # Limpieza de datos nulos
            df.fillna('', inplace=True)  # Rellena nulos con cadenas vacías

            # Asegurarse de que los tipos de datos sean compatibles
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(str)  # Convierte a cadena si la columna es float64

            # Convertir columnas datetime a string
            for col in df.select_dtypes(include=['datetime']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')  # Formato de fecha y hora en string

            return df
        except Exception as e:
            raise ValueError(f"Error al procesar el archivo .xls: {str(e)}")

    @staticmethod
    def insertar_datos_mongo(datos_json: list):
        try:
            # Conectar a la base de datos
            client = MongoClient(settings.mongo_uri)
            db = client["sistemaPredic"]
            collection = db["Datos_dengue"]

            # Insertar los datos en MongoDB
            result = collection.insert_many(datos_json)  # Insertar múltiples documentos

            # Convertir ObjectId a cadena antes de devolverlos
            inserted_ids = [str(id) for id in result.inserted_ids]
            return inserted_ids  # Retorna los IDs como cadenas
        except Exception as e:
            raise ValueError(f"Error al insertar los datos en MongoDB: {str(e)}")
