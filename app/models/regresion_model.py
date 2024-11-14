from sklearn.linear_model import Ridge
from joblib import dump, load

class RegresionModel:
    def __init__(self):
        self.model = Ridge(alpha=1.0)  # Regularización L2

    def entrenar(self, X, y):
        self.model.fit(X, y)
        dump(self.model, 'regresion_model.pkl')

    def predecir(self, X):
        self.model = load('regresion_model.pkl')
        return self.model.predict(X)
