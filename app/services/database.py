from pymongo import MongoClient
from app.config import settings

def get_database():
    # Conectar a MongoDB usando la URI completa (con base de datos especificada directamente)
    client = MongoClient(settings.mongo_uri)
    
    # Seleccionamos directamente la base de datos
    db = client["sistemaPredic"]  # Aquí especificamos el nombre de la base de datos
    
    # Nombre de la colección
    collection = db["nombre_coleccion"]
    return collection
