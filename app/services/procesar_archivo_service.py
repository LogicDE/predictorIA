import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
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
        
    @staticmethod
    def cargar_datos_mongo():
        try:
            # Conectar a MongoDB
            client = MongoClient(settings.mongo_uri)
            db = client["sistemaPredic"]
            collection = db["Datos_dengue"]

            # Obtener todos los documentos y convertir a DataFrame
            datos = list(collection.find())
            df = pd.DataFrame(datos)

            # Eliminar el campo '_id' ya que no lo necesitamos
            df = df.drop(columns=['_id'], errors='ignore')

            # Validar si 'fec_not' y 'ini_sin_' son fechas, convertirlas y crear variables derivadas
            if 'fec_not' in df.columns:
                df['fec_not'] = pd.to_datetime(df['fec_not'], errors='coerce')
                df = df.dropna(subset=['fec_not'])  # Eliminar filas con fechas nulas
                df['año'] = df['fec_not'].dt.year
                df['mes'] = df['fec_not'].dt.month
                df['día'] = df['fec_not'].dt.day
                df['día_semana'] = df['fec_not'].dt.weekday

            if 'ini_sin_' in df.columns:
                df['ini_sin_'] = pd.to_datetime(df['ini_sin_'], errors='coerce')
                df = df.dropna(subset=['ini_sin_'])  # Eliminar filas con fechas nulas

            # Manejo de variables categóricas
            categ_cols = ['sexo_', 'tip_ss_', 'nom_eve', 'nmun_proce', 'localidad_']
            le = LabelEncoder()
            for col in categ_cols:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))

            # Filtrar las columnas necesarias (manteniendo todas disponibles para análisis general)
            columnas_finales = [
                'cod_eve', 'fec_not', 'semana', 'edad_', 'sexo_', 'cod_dpto_o', 'cod_mun_o', 'area_',
                'localidad_', 'cen_pobla_', 'tip_ss_', 'estrato_', 'ini_sin_', 'tip_cas_', 'ajuste_', 'fiebre',
                'cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito', 'diarrea',
                'somnolenci', 'hipotensio', 'hepatomeg', 'hem_mucosa', 'hipotermia', 'aum_hemato', 'caida_plaq',
                'acum_liqui', 'extravasac', 'hemorr_hem', 'choque', 'daño_organ', 'clasfinal', 'conducta', 'nom_eve',
                'nmun_proce', 'año', 'mes', 'día', 'día_semana'
            ]
            df = df[columnas_finales]

            return df
        except Exception as e:
            raise ValueError(f"Error al cargar y procesar datos de MongoDB: {str(e)}")
