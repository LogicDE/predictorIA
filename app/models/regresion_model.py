from sklearn.linear_model import LinearRegression
from joblib import dump, load

class RegresionModel:
    def __init__(self):
        self.model = LinearRegression()

    def entrenar(self, X, y):
        self.model.fit(X, y)
        dump(self.model, 'regresion_model.pkl')

    def predecir(self, X):
        self.model = load('regresion_model.pkl')
        return self.model.predict(X)
