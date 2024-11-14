from pydantic_settings import BaseSettings
from pydantic import ValidationError
from dotenv import load_dotenv
import os

# Cargar el archivo .env con la ruta completa
load_dotenv("C:/Users/themo/OneDrive/Documents/predictorIA/app/.env")

class Settings(BaseSettings):
    mongo_uri: str

    class Config:
        env_file = "C:/Users/themo/OneDrive/Documents/predictorIA/app/.env"

# Intentar cargar las configuraciones y capturar cualquier error de validación
try:
    settings = Settings()
    print("Configuración cargada exitosamente.")
    print("MONGO_URI:", settings.mongo_uri)  # Verificar si se cargó correctamente
except ValidationError as e:
    print("Error al cargar configuraciones:", e)
    raise
