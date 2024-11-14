from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np

class RedNeuronalModel:
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def entrenar(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        self.model.save('red_neuronal_model.h5')

    def predecir(self, X):
        self.model = load_model('red_neuronal_model.h5')
        return self.model.predict(X)
